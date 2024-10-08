mod async_waiter;
mod asyncs;
mod dataloader;
mod host_protocol;
mod stateprovider;
mod traintracker;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::trait_def::*;
use dataloader::*;

use futures::FutureExt;

struct PanicInfo {
    backtrace: String,
    location: Option<(String, u32, u32)>,
    thread_name: Option<String>,
}

lazy_static::lazy_static! {
    static ref PANIC_INFO: Arc<Mutex<Option<PanicInfo>>> = Arc::new(Mutex::new(None));
}

fn panic_hook(msg: &std::panic::PanicInfo) {
    *PANIC_INFO.lock().unwrap() = Some(PanicInfo {
        backtrace: std::backtrace::Backtrace::force_capture().to_string(),
        location: msg
            .location()
            .map(|x| (x.file().to_owned(), x.line(), x.column())),
        thread_name: std::thread::current().name().map(|x| x.to_owned()),
    });
}

fn format_panic(e: Box<dyn std::any::Any>) -> String {
    let panic_info = PANIC_INFO
        .lock()
        .unwrap()
        .take()
        .unwrap_or_else(|| PanicInfo {
            backtrace: "<Backtrace not found>".to_string(),
            location: None,
            thread_name: None,
        });
    let location = if let Some(location) = panic_info.location {
        format!(" at {}:{}:{}", location.0, location.1, location.2)
    } else {
        "".to_string()
    };
    let thread_name = if let Some(thread_name) = panic_info.thread_name {
        format!("Thread '{thread_name}' p")
    } else {
        "P".to_string()
    };
    let e = match e.downcast::<&str>() {
        Ok(val) => {
            return format!(
                "{thread_name}anicked{location}:\n{val}\nBacktrace:\n{}",
                panic_info.backtrace,
            );
        }
        Err(e) => e,
    };
    if let Ok(val) = e.downcast::<String>() {
        return format!(
            "{thread_name}anicked{location}:\n{val}\nBacktrace:\n{}",
            panic_info.backtrace,
        );
    }
    format!(
        "{thread_name}anicked{location}:\n<unknown panic info>.\nBacktrace:\n{}",
        panic_info.backtrace,
    )
}

struct InstantiatedModelWaiter<I: InstantiatedBinary> {
    waiter: async_waiter::AsyncWaiter<I>,
    dispose_tx: asyncs::oneshot::Sender<()>,
}

struct Runner<M: ModelBinary> {
    sender: host_protocol::Sender,
    data_loader_manager: DataLoaderManager,
    instantiated_models: Arc<Mutex<HashMap<String, InstantiatedModelWaiter<M::Instantiated>>>>,
    training_sessions: Arc<Mutex<HashMap<String, async_waiter::AsyncWaiterProvider<()>>>>,
}

impl<M: ModelBinary> Clone for Runner<M> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            data_loader_manager: self.data_loader_manager.clone(),
            instantiated_models: self.instantiated_models.clone(),
            training_sessions: self.training_sessions.clone(),
        }
    }
}

impl<M: ModelBinary + Send + Sync + 'static> Runner<M>
where
    M::Instantiated: Send + Sync,
{
    fn create_data_loader(
        &self,
        dataset: String,
        size: u32,
        total_byte_size: u64,
    ) -> impl DataLoaderBinary + 'static {
        let (data_loader, fut) =
            self.data_loader_manager
                .create_data_loader(dataset, size, total_byte_size);
        asyncs::spawn(fut);
        data_loader
    }

    fn create_state_loader(
        &self,
        dataset: String,
        total_byte_size: u64,
    ) -> impl StateLoader + 'static {
        let (data_loader, fut) = self
            .data_loader_manager
            .create_state_loader(dataset, total_byte_size);
        asyncs::spawn(fut);
        data_loader
    }

    async fn handle_command(
        &self,
        command: host_protocol::CommandMessage,
    ) -> Option<(String, host_protocol::ResultMessage, Vec<bytes::Bytes>)> {
        match command {
            host_protocol::CommandMessage::CallCreateModelState {
                id,
                params,
                other_models,
            } => {
                let error = match std::panic::AssertUnwindSafe(M::create_model_state(
                    crate::trait_def::CreateModelStateOptions {
                        params: params
                            .into_iter()
                            .map(|x| {
                                (
                                    x.name,
                                    self.create_data_loader(x.dataset, x.amount, x.total_byte_size),
                                )
                            })
                            .collect(),
                        state_provider: stateprovider::create_state_provider(
                            &id,
                            self.sender.clone(),
                        ),
                        other_models: other_models
                            .into_iter()
                            .map(|other_model| {
                                (
                                    other_model.id,
                                    crate::trait_def::OtherModelWithState {
                                        mount_path: other_model.mount_path,
                                        state: other_model
                                            .state
                                            .into_iter()
                                            .map(|param| {
                                                (
                                                    param.name,
                                                    self.create_state_loader(
                                                        param.dataset,
                                                        param.total_byte_size,
                                                    ),
                                                )
                                            })
                                            .collect(),
                                    },
                                )
                            })
                            .collect(),
                    },
                ))
                .catch_unwind()
                .await
                {
                    Ok(()) => None,
                    Err(e) => Some(host_protocol::CallCreateModelStateError::Exception {
                        details: Some(format_panic(e)),
                    }),
                };
                Some((
                    id,
                    host_protocol::ResultMessage::CallCreateModelState { error },
                    vec![],
                ))
            }
            host_protocol::CommandMessage::CallInstantiateModel {
                id,
                instantiated_model_id,
                state,
                other_models,
            } => {
                let (waiter, provider) = async_waiter::AsyncWaiter::<M::Instantiated>::new();

                let (dispose_tx, dispose_rx) = asyncs::oneshot::channel();

                {
                    let mut instantiated_models = self.instantiated_models.lock().unwrap();
                    instantiated_models.insert(
                        instantiated_model_id,
                        InstantiatedModelWaiter { waiter, dispose_tx },
                    );
                }

                let mut provider = Some(provider);
                let drop_fut = async {
                    dispose_rx.await.unwrap();
                    provider = None;
                    Ok(())
                };
                let instantiate_fut = async {
                    let res = std::panic::AssertUnwindSafe(M::instantiate_model(
                        crate::trait_def::InstantiateModelOptions {
                            state: state
                                .into_iter()
                                .map(|x| {
                                    (
                                        x.name,
                                        self.create_state_loader(x.dataset, x.total_byte_size),
                                    )
                                })
                                .collect(),
                            other_models: other_models
                                .into_iter()
                                .map(|other_model| {
                                    (
                                        other_model.id,
                                        crate::trait_def::OtherModel {
                                            mount_path: other_model.mount_path,
                                        },
                                    )
                                })
                                .collect(),
                        },
                    ))
                    .catch_unwind()
                    .await;
                    Err::<(), _>(res)
                };

                let res = futures::try_join!(drop_fut, instantiate_fut).unwrap_err();

                let error = match res {
                    Ok(model) => {
                        if let Some(provider) = provider {
                            provider.provide(model);
                        }
                        None
                    }
                    Err(e) => Some(host_protocol::CallInstantiateModelError::Exception {
                        details: Some(format_panic(e)),
                    }),
                };
                Some((
                    id,
                    host_protocol::ResultMessage::CallInstantiateModel { error },
                    vec![],
                ))
            }
            host_protocol::CommandMessage::CallDisposeInstantiatedModel {
                instantiated_model_id,
            } => {
                let mut instantiated_models = self.instantiated_models.lock().unwrap();
                let disposed = instantiated_models.remove(&instantiated_model_id);
                drop(instantiated_models);
                if let Some(disposed) = disposed {
                    disposed.dispose_tx.send(()).ok();
                }
                None
            }
            host_protocol::CommandMessage::CallTrain {
                id,
                training_session_id,
                instantiated_model_id,
                params,
            } => {
                let instantiated = {
                    let instantiated_models = self.instantiated_models.lock().unwrap();
                    instantiated_models
                        .get(&instantiated_model_id)
                        .map(|x| x.waiter.clone())
                };
                let instantiated = if let Some(instantiated) = instantiated {
                    instantiated.get().await
                } else {
                    None
                };

                let error = match instantiated {
                    Some(instantiated) => {
                        let (train_tracker, cancel_tx) = traintracker::create_train_tracker(
                            self.sender.clone(),
                            &training_session_id,
                        );

                        {
                            let mut training_sessions = self.training_sessions.lock().unwrap();
                            training_sessions.insert(training_session_id.clone(), cancel_tx);
                        }

                        let res = std::panic::AssertUnwindSafe(
                            instantiated.as_ref().train(crate::trait_def::TrainOptions {
                                params: params
                                    .into_iter()
                                    .map(|x| {
                                        (
                                            x.name,
                                            self.create_data_loader(
                                                x.dataset,
                                                x.amount,
                                                x.total_byte_size,
                                            ),
                                        )
                                    })
                                    .collect(),
                                tracker: train_tracker,
                            }),
                        )
                        .catch_unwind()
                        .await;

                        let mut training_sessions = self.training_sessions.lock().unwrap();
                        training_sessions.remove(&training_session_id);
                        drop(training_sessions);

                        match res {
                            Ok(()) => None,
                            Err(e) => Some(host_protocol::CallTrainError::Exception {
                                details: Some(format_panic(e)),
                            }),
                        }
                    }
                    None => Some(host_protocol::CallTrainError::InstantiatedModelNotFound),
                };
                Some((
                    id,
                    host_protocol::ResultMessage::CallTrain { error },
                    vec![],
                ))
            }
            host_protocol::CommandMessage::CallCancelTrain {
                training_session_id,
            } => {
                let mut training_sessions = self.training_sessions.lock().unwrap();
                let cancel_tx = training_sessions.remove(&training_session_id);
                drop(training_sessions);

                if let Some(cancel_tx) = cancel_tx {
                    cancel_tx.provide(());
                }
                None
            }
            host_protocol::CommandMessage::CallEvaluate {
                id,
                instantiated_model_id,
                params,
                expected_output_types,
            } => {
                let instantiated = {
                    let instantiated_models = self.instantiated_models.lock().unwrap();
                    instantiated_models
                        .get(&instantiated_model_id)
                        .map(|x| x.waiter.clone())
                };
                let instantiated = if let Some(instantiated) = instantiated {
                    instantiated.get().await
                } else {
                    None
                };
                let (outputs, error, data) = match instantiated {
                    Some(instantiated) => {
                        let res = std::panic::AssertUnwindSafe(
                            instantiated
                                .as_ref()
                                .evaluate(crate::trait_def::EvaluateOptions {
                                    params: params
                                        .into_iter()
                                        .map(|x| {
                                            (
                                                x.name,
                                                self.create_data_loader(
                                                    x.dataset,
                                                    x.amount,
                                                    x.total_byte_size,
                                                ),
                                            )
                                        })
                                        .collect(),
                                    expected_output_types: expected_output_types
                                        .into_iter()
                                        .map(|x| (x.name, x.rules))
                                        .collect(),
                                }),
                        )
                        .catch_unwind()
                        .await;

                        match res {
                            Ok(val) => {
                                let (outputs, data): (Vec<_>, Vec<_>) = val
                                    .into_iter()
                                    .map(|x| {
                                        (
                                            host_protocol::EvaluateOutput {
                                                name: x.name,
                                                byte_sizes: x
                                                    .data
                                                    .iter()
                                                    .map(|x| x.len() as u64)
                                                    .collect(),
                                            },
                                            x.data,
                                        )
                                    })
                                    .unzip();
                                (
                                    Some(outputs),
                                    None,
                                    vec![data
                                        .into_iter()
                                        .flatten()
                                        .collect::<Vec<_>>()
                                        .concat()
                                        .into()],
                                )
                            }
                            Err(e) => (
                                None,
                                Some(host_protocol::CallEvaluateError::Exception {
                                    details: Some(format_panic(e)),
                                }),
                                vec![],
                            ),
                        }
                    }
                    None => (
                        None,
                        Some(host_protocol::CallEvaluateError::InstantiatedModelNotFound),
                        vec![],
                    ),
                };
                Some((
                    id,
                    host_protocol::ResultMessage::CallEvaluate { outputs, error },
                    data,
                ))
            }
            host_protocol::CommandMessage::CallGetModelState {
                id,
                instantiated_model_id,
            } => {
                let instantiated = {
                    let instantiated_models = self.instantiated_models.lock().unwrap();
                    instantiated_models
                        .get(&instantiated_model_id)
                        .map(|x| x.waiter.clone())
                };
                let instantiated = if let Some(instantiated) = instantiated {
                    instantiated.get().await
                } else {
                    None
                };
                let error = match instantiated {
                    Some(instantiated) => {
                        match std::panic::AssertUnwindSafe(instantiated.as_ref().get_model_state(
                            crate::trait_def::GetModelStateOptions {
                                state_provider: stateprovider::create_state_provider(
                                    &id,
                                    self.sender.clone(),
                                ),
                            },
                        ))
                        .catch_unwind()
                        .await
                        {
                            Ok(()) => None,
                            Err(e) => Some(host_protocol::CallGetModelStateError::Exception {
                                details: Some(format_panic(e)),
                            }),
                        }
                    }
                    None => Some(host_protocol::CallGetModelStateError::InstantiatedModelNotFound),
                };
                Some((
                    id,
                    host_protocol::ResultMessage::CallGetModelState { error },
                    vec![],
                ))
            }
        }
    }

    async fn run<R: asyncs::AsyncRead + Unpin>(&self, mut reader: R) {
        loop {
            let runner = self.clone();
            match host_protocol::read_message_from_host(&mut reader)
                .await
                .expect("Failed to read incoming message from host")
            {
                host_protocol::MessageFromHost::Command(cmd) => {
                    asyncs::spawn(async move {
                        let Some((id, result, blobs_output)) = runner.handle_command(cmd).await
                        else {
                            return;
                        };

                        runner.sender.send_result(id, result, blobs_output).await;
                    });
                }
                host_protocol::MessageFromHost::ProvideData(request_id, data) => {
                    self.data_loader_manager.provide_data(request_id, data)
                }
            }
        }
    }
}

pub async fn run_model<M: ModelBinary + Send + Sync + 'static>()
where
    M::Instantiated: Send + Sync,
{
    std::panic::set_hook(Box::new(panic_hook));

    let mut connection = asyncs::UnixStream::connect(
        std::env::var("IPC_PATH").expect("Expected an environment variable \"IPC_PATH\""),
    )
    .await
    .expect("Failed to connect to Decthings host over unix socket");
    let (reader, writer) = asyncs::unix_split(&mut connection);

    let writer = asyncs::BufWriter::new(writer);
    let reader = asyncs::BufReader::new(reader);

    let (sender, sender_fut) = host_protocol::Sender::new(writer);

    let runner = Runner::<M> {
        sender: sender.clone(),
        data_loader_manager: DataLoaderManager::new(sender.clone()),
        instantiated_models: Arc::new(Mutex::new(HashMap::new())),
        training_sessions: Arc::new(Mutex::new(HashMap::new())),
    };
    let runner = &runner;

    futures::join!(
        async {
            sender
                .send_event::<String>(
                    host_protocol::EventMessage::ModelSessionInitialized {},
                    vec![],
                )
                .await;
        },
        sender_fut,
        runner.run(reader),
    );
}
