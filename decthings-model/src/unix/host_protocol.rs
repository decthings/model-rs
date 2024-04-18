use std::future::Future;

use serde::ser::SerializeSeq;

use super::asyncs::{AsyncReadExt, AsyncWriteExt};

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Param {
    pub name: String,
    pub dataset: String,
    pub amount: u32,
    pub total_byte_size: u64,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OtherModelWithState {
    pub id: String,
    pub mount_path: String,
    pub state: Vec<Param>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OtherModel {
    pub id: String,
    pub mount_path: String,
}

#[allow(clippy::enum_variant_names)]
#[derive(serde::Deserialize)]
#[serde(tag = "method", content = "params", rename_all = "camelCase")]
pub enum CommandMessage {
    #[serde(rename_all = "camelCase")]
    CallCreateModelState {
        id: String,
        params: Vec<Param>,
        other_models: Vec<OtherModelWithState>,
    },
    #[serde(rename_all = "camelCase")]
    CallInstantiateModel {
        id: String,
        instantiated_model_id: String,
        state: Vec<Param>,
        other_models: Vec<OtherModel>,
    },
    #[serde(rename_all = "camelCase")]
    CallDisposeInstantiatedModel { instantiated_model_id: String },
    #[serde(rename_all = "camelCase")]
    CallTrain {
        id: String,
        training_session_id: String,
        instantiated_model_id: String,
        params: Vec<Param>,
    },
    #[serde(rename_all = "camelCase")]
    CallCancelTrain { training_session_id: String },
    #[serde(rename_all = "camelCase")]
    CallEvaluate {
        id: String,
        instantiated_model_id: String,
        params: Vec<Param>,
    },
    #[serde(rename_all = "camelCase")]
    CallGetModelState {
        id: String,
        instantiated_model_id: String,
    },
}

#[derive(serde::Serialize)]
#[serde(rename_all = "snake_case", tag = "code")]
pub enum CallCreateModelStateError {
    Exception {
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
}

#[derive(serde::Serialize)]
#[serde(rename_all = "snake_case", tag = "code")]
pub enum CallInstantiateModelError {
    Exception {
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
}

#[derive(serde::Serialize)]
#[serde(rename_all = "snake_case", tag = "code")]
pub enum CallTrainError {
    Exception {
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    InstantiatedModelNotFound,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EvaluateOutput {
    pub name: String,
    pub byte_sizes: Vec<u64>,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "snake_case", tag = "code")]
pub enum CallEvaluateError {
    Exception {
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    InstantiatedModelNotFound,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "snake_case", tag = "code")]
pub enum CallGetModelStateError {
    Exception {
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    InstantiatedModelNotFound,
}

#[allow(clippy::enum_variant_names)]
#[derive(serde::Serialize)]
#[serde(untagged, rename_all = "camelCase")]
pub enum ResultMessage {
    #[serde(rename_all = "camelCase")]
    CallCreateModelState {
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<CallCreateModelStateError>,
    },
    #[serde(rename_all = "camelCase")]
    CallInstantiateModel {
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<CallInstantiateModelError>,
    },
    #[serde(rename_all = "camelCase")]
    CallTrain {
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<CallTrainError>,
    },
    #[serde(rename_all = "camelCase")]
    CallEvaluate {
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<CallEvaluateError>,
        #[serde(skip_serializing_if = "Option::is_none")]
        outputs: Option<Vec<EvaluateOutput>>,
    },
    #[serde(rename_all = "camelCase")]
    CallGetModelState {
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<CallGetModelStateError>,
    },
}

fn serialize_asref_str_seq<S: serde::Serializer, T: AsRef<str>>(
    values: &&[T],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut seq = serializer.serialize_seq(Some(values.len()))?;
    for value in *values {
        seq.serialize_element(value.as_ref())?;
    }
    seq.end()
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
#[serde(tag = "event", content = "params")]
pub enum EventMessage<'a, S: AsRef<str>> {
    #[serde(rename_all = "camelCase")]
    ModelSessionInitialized {},
    #[serde(rename_all = "camelCase")]
    TrainingProgress {
        training_session_id: &'a str,
        progress: f32,
    },
    #[serde(rename_all = "camelCase")]
    TrainingMetrics {
        training_session_id: &'a str,
        #[serde(serialize_with = "serialize_asref_str_seq")]
        names: &'a [S],
    },
    #[serde(rename_all = "camelCase")]
    ProvideStateData {
        command_id: &'a str,
        #[serde(serialize_with = "serialize_asref_str_seq")]
        names: &'a [S],
    },
}

#[derive(Debug, serde::Serialize)]
#[serde(tag = "event", rename_all = "camelCase")]
pub enum DataEvent<'a> {
    #[serde(rename_all = "camelCase")]
    RequestData {
        dataset: &'a str,
        request_id: u32,
        start_index: u32,
        amount: u32,
    },
    #[serde(rename_all = "camelCase")]
    Shuffle { datasets: &'a [&'a str] },
}

pub enum MessageFromHost {
    Command(CommandMessage),
    ProvideData(u32, Vec<bytes::Bytes>),
}

pub async fn read_message_from_host(
    mut reader: impl super::asyncs::AsyncRead + Unpin,
) -> Result<MessageFromHost, std::io::Error> {
    let first_byte = super::asyncs::read_u8(&mut reader).await?;
    if first_byte == 0 {
        // RPC
        let segment_length = super::asyncs::read_u64(&mut reader).await? as usize;
        let mut buf = vec![0; segment_length as usize];
        reader.read_exact(&mut buf).await?;
        Ok(MessageFromHost::Command(
            serde_json::from_slice(&buf).expect("Failed to parse incoming message from host"),
        ))
    } else {
        // Provide data
        let request_id = super::asyncs::read_u32(&mut reader).await?;
        let num_segments = super::asyncs::read_u32(&mut reader).await?;
        let mut data = Vec::with_capacity(num_segments as usize);
        for _ in 0..num_segments {
            let segment_length = super::asyncs::read_u64(&mut reader).await? as usize;
            let mut buf = vec![0; segment_length as usize];
            reader.read_exact(&mut buf).await?;
            data.push(buf.into());
        }
        Ok(MessageFromHost::ProvideData(request_id, data))
    }
}

enum MessageToHost {
    ResultOrEvent(Vec<u8>, Vec<bytes::Bytes>),
    DataEvent(Vec<u8>),
}

#[derive(Clone)]
pub struct Sender {
    tx: super::asyncs::Sender<MessageToHost>,
}

impl Sender {
    pub fn new<W: super::asyncs::AsyncWrite + Unpin>(
        mut writer: W,
    ) -> (Self, impl Future<Output = ()>) {
        let (tx, mut rx) = super::asyncs::channel(1);
        (Self { tx }, async move {
            async {
                while let Some(msg) = super::asyncs::channel_recv(&mut rx).await {
                    match msg {
                        MessageToHost::ResultOrEvent(msg, additional_segments) => {
                            super::asyncs::write_u8(&mut writer, 0).await?;
                            super::asyncs::write_u32(
                                &mut writer,
                                additional_segments.len().try_into().unwrap(),
                            )
                            .await?;
                            super::asyncs::write_u64(&mut writer, msg.len() as u64).await?;
                            writer.write_all(&msg).await?;
                            drop(msg);
                            for additional in additional_segments {
                                super::asyncs::write_u64(
                                    &mut writer,
                                    additional.as_ref().len().try_into().unwrap(),
                                )
                                .await?;
                                writer.write_all(additional.as_ref()).await?;
                            }
                            super::asyncs::write_u8(&mut writer, 1).await?;
                        }
                        MessageToHost::DataEvent(msg) => {
                            super::asyncs::write_u8(&mut writer, 1).await?;
                            super::asyncs::write_u64(&mut writer, msg.len().try_into().unwrap())
                                .await?;
                            writer.write_all(&msg).await?;
                        }
                    }
                    writer.flush().await?;
                }

                Ok::<(), std::io::Error>(())
            }
            .await
            .expect("Failed to communicate with host");
        })
    }

    pub async fn send_result(
        &self,
        id: String,
        result: ResultMessage,
        additional_segments: Vec<bytes::Bytes>,
    ) {
        #[derive(serde::Serialize)]
        struct ResultMessageWithId {
            result: ResultMessage,
            id: String,
        }
        let msg = serde_json::to_vec(&ResultMessageWithId { id, result }).unwrap();
        self.tx
            .send(MessageToHost::ResultOrEvent(msg, additional_segments))
            .await
            .map_err(|_| ())
            .unwrap();
    }

    pub async fn send_event<S: AsRef<str>>(
        &self,
        event: EventMessage<'_, S>,
        additional_segments: Vec<bytes::Bytes>,
    ) {
        let msg = serde_json::to_vec(&event).unwrap();
        self.tx
            .send(MessageToHost::ResultOrEvent(msg, additional_segments))
            .await
            .map_err(|_| ())
            .unwrap();
    }

    pub async fn send_data_event(&self, data_event: DataEvent<'_>) {
        let msg = serde_json::to_vec(&data_event).unwrap();
        self.tx
            .send(MessageToHost::DataEvent(msg))
            .await
            .map_err(|_| ())
            .unwrap();
    }
}
