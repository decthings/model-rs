use std::sync::{Arc, RwLock};

enum ValueOrQueue<T> {
    Value(Arc<T>),
    Queue(Vec<super::asyncs::oneshot::Sender<Arc<T>>>),
}

pub struct AsyncWaiterProvider<T> {
    value_or_queue: Arc<RwLock<ValueOrQueue<T>>>,
}

impl<T> AsyncWaiterProvider<T> {
    pub fn provide(self, value: T) {
        let val = Arc::new(value);
        let waiting = std::mem::replace(
            &mut *self.value_or_queue.write().unwrap(),
            ValueOrQueue::Value(Arc::clone(&val)),
        );
        match waiting {
            ValueOrQueue::Value(_) => unreachable!(),
            ValueOrQueue::Queue(queue) => {
                for tx in queue {
                    tx.send(Arc::clone(&val)).ok();
                }
            }
        }
    }
}

pub struct AsyncWaiter<T> {
    value_or_queue: Arc<RwLock<ValueOrQueue<T>>>,
}

impl<T> Clone for AsyncWaiter<T> {
    fn clone(&self) -> Self {
        Self {
            value_or_queue: Arc::clone(&self.value_or_queue),
        }
    }
}

impl<T> AsyncWaiter<T> {
    pub fn new() -> (Self, AsyncWaiterProvider<T>) {
        let queue = Arc::new(RwLock::new(ValueOrQueue::Queue(vec![])));

        (
            Self {
                value_or_queue: Arc::clone(&queue),
            },
            AsyncWaiterProvider {
                value_or_queue: queue,
            },
        )
    }

    fn inner_get(&self) -> Result<Arc<T>, super::asyncs::oneshot::Receiver<Arc<T>>> {
        let locked = self.value_or_queue.read().unwrap();
        match &*locked {
            ValueOrQueue::Queue(_) => {
                drop(locked);
                let mut locked = self.value_or_queue.write().unwrap();
                match &mut *locked {
                    ValueOrQueue::Queue(queue) => {
                        let (tx, rx) = super::asyncs::oneshot::channel();
                        queue.push(tx);
                        Err(rx)
                    }
                    ValueOrQueue::Value(val) => {
                        let cloned = Arc::clone(val);
                        Ok(cloned)
                    }
                }
            }
            ValueOrQueue::Value(val) => {
                let cloned = Arc::clone(val);
                Ok(cloned)
            }
        }
    }

    pub async fn get(&self) -> Option<Arc<T>> {
        match self.inner_get() {
            Ok(val) => Some(val),
            Err(rx) => rx.await.ok(),
        }
    }
}
