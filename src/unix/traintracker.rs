use crate::*;
use futures::future::BoxFuture;

struct TrainTrackerImpl<'a> {
    sender: super::host_protocol::Sender,
    cancel_waiter: super::async_waiter::AsyncWaiter<()>,
    training_session_id: &'a str,
}

impl<'a> TrainTrackerBinary for TrainTrackerImpl<'a> {
    fn wait_for_cancelled(&self) -> BoxFuture<'_, ()> {
        Box::pin(async {
            self.cancel_waiter.get().await;
        })
    }

    fn progress(&self, progress: f32) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            self.sender
                .send_event::<String>(
                    super::host_protocol::EventMessage::TrainingProgress {
                        training_session_id: self.training_session_id,
                        progress,
                    },
                    vec![],
                )
                .await;
        })
    }

    fn metrics<'b>(
        &'b self,
        metrics: &'b [MetricBinary<impl AsRef<str> + Sync + 'b>],
    ) -> BoxFuture<'b, ()> {
        Box::pin(async {
            self.sender
                .send_event::<_>(
                    super::host_protocol::EventMessage::TrainingMetrics {
                        training_session_id: self.training_session_id,
                        names: &metrics.iter().map(|x| &x.name).collect::<Vec<_>>(),
                    },
                    metrics.iter().map(|x| x.data.clone()).collect(),
                )
                .await;
        })
    }
}

pub(super) fn create_train_tracker(
    sender: super::host_protocol::Sender,
    training_session_id: &str,
) -> (
    impl TrainTrackerBinary + '_,
    super::async_waiter::AsyncWaiterProvider<()>,
) {
    let (waiter, provider) = super::async_waiter::AsyncWaiter::new();
    (
        TrainTrackerImpl {
            sender,
            cancel_waiter: waiter,
            training_session_id,
        },
        provider,
    )
}
