use std::{
    collections::HashMap,
    future::Future,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use crate::*;
use futures::future::BoxFuture;

struct RequestData {
    start_index: u32,
    amount: u32,
    cb: super::asyncs::oneshot::Sender<Vec<bytes::Bytes>>,
}

pub struct DataLoaderImpl<'a> {
    _phantom: PhantomData<&'a ()>,
    sender: super::host_protocol::Sender,
    request_data_tx: super::asyncs::Sender<RequestData>,
    dataset: String,
    size: u32,
    total_byte_size: u64,
    position: u32,
}

impl<'a> DataLoaderBinary for DataLoaderImpl<'a> {
    fn total_byte_size(&self) -> u64 {
        self.total_byte_size
    }

    fn shuffle_in_group<'b>(&'b self, others: &'b [&'b Self]) -> BoxFuture<'b, ()> {
        let datasets: Vec<_> = [self]
            .iter()
            .chain(others)
            .map(|x| x.dataset.as_str())
            .collect();
        Box::pin(async move {
            self.sender
                .send_data_event(super::host_protocol::DataEvent::Shuffle {
                    datasets: &datasets,
                })
                .await;
        })
    }

    fn size(&self) -> u32 {
        self.size
    }

    fn position(&self) -> u32 {
        self.position
    }

    fn set_position(&mut self, position: u32) {
        if position >= self.size {
            panic!(
                "DataLoader: Cannot set the the position to a value greater than or equal to the data size. The data size was {}, and position {} was attempted to be set.",
                self.size, position
            );
        }
        self.position = position;
    }

    fn remaining(&self) -> u32 {
        self.size - self.position
    }

    fn next(&mut self, mut amount: u32) -> BoxFuture<'_, Vec<bytes::Bytes>> {
        Box::pin(async move {
            amount = amount.min(DataLoaderBinary::remaining(self));

            if amount == 0 {
                return vec![];
            }

            let prev_position = DataLoaderBinary::position(self);
            self.position += amount;

            let (tx, rx) = super::asyncs::oneshot::channel();

            self.request_data_tx
                .send(RequestData {
                    start_index: prev_position,
                    amount,
                    cb: tx,
                })
                .await
                .map_err(|_| ())
                .unwrap();

            rx.await.unwrap()
        })
    }
}

impl<'a> crate::WeightsLoader for DataLoaderImpl<'a> {
    fn byte_size(&self) -> u64 {
        crate::DataLoaderBinary::total_byte_size(self)
    }

    fn read(&mut self) -> BoxFuture<'_, bytes::Bytes> {
        Box::pin(async {
            crate::DataLoaderBinary::set_position(self, 0);
            crate::DataLoaderBinary::next(self, 1).await.remove(0)
        })
    }
}

struct Requests {
    waiting: HashMap<u32, super::asyncs::oneshot::Sender<Vec<bytes::Bytes>>>,
    id_counter: u32,
}

#[derive(Clone)]
pub(super) struct DataLoaderManager {
    sender: super::host_protocol::Sender,
    requests: Arc<Mutex<Requests>>,
}

impl DataLoaderManager {
    pub fn new(sender: super::host_protocol::Sender) -> Self {
        Self {
            sender,
            requests: Arc::new(Mutex::new(Requests {
                waiting: HashMap::new(),
                id_counter: 0,
            })),
        }
    }

    pub fn provide_data(&self, request_id: u32, data: Vec<bytes::Bytes>) {
        let mut requests = self.requests.lock().unwrap();
        let waiting = requests.waiting.remove(&request_id);

        if let Some(waiting) = waiting {
            waiting.send(data).ok();
        }
    }

    fn do_create_data_loader(
        &self,
        dataset: String,
        size: u32,
        total_byte_size: u64,
    ) -> (
        impl DataLoaderBinary + WeightsLoader + 'static,
        impl Future<Output = ()> + Send + 'static,
    ) {
        let (tx, mut rx) = super::asyncs::channel(1);

        let sender = self.sender.clone();
        let requests = self.requests.clone();

        (
            DataLoaderImpl {
                _phantom: PhantomData,
                sender: sender.clone(),
                request_data_tx: tx,
                dataset: dataset.clone(),
                size,
                total_byte_size,
                position: 0,
            },
            async move {
                while let Some(request) = super::asyncs::channel_recv(&mut rx).await {
                    let request_id = {
                        let mut requests = requests.lock().unwrap();

                        let request_id = requests.id_counter;
                        requests.id_counter += 1;
                        requests.waiting.insert(request_id, request.cb);

                        request_id
                    };

                    sender
                        .send_data_event(super::host_protocol::DataEvent::RequestData {
                            request_id,
                            dataset: &dataset,
                            start_index: request.start_index,
                            amount: request.amount,
                        })
                        .await;
                }
            },
        )
    }

    pub fn create_data_loader(
        &self,
        dataset: String,
        size: u32,
        total_byte_size: u64,
    ) -> (
        impl DataLoaderBinary + 'static,
        impl Future<Output = ()> + Send + 'static,
    ) {
        self.do_create_data_loader(dataset, size, total_byte_size)
    }

    pub fn create_weights_loader(
        &self,
        dataset: String,
        byte_size: u64,
    ) -> (
        impl WeightsLoader + 'static,
        impl Future<Output = ()> + Send + 'static,
    ) {
        self.do_create_data_loader(dataset, 1, byte_size)
    }
}
