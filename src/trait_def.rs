use std::collections::HashMap;

use futures::future::BoxFuture;

use decthings_api::tensor::{DecthingsTensor, OwnedDecthingsTensor};

#[derive(Clone, Debug)]
pub struct EvaluateOutput {
    pub name: String,
    pub data: Vec<OwnedDecthingsTensor>,
}

#[cfg_attr(target_family = "unix", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct EvaluateOutputBinary {
    pub name: String,
    #[cfg_attr(target_family = "unix", serde(skip_serializing, skip_deserializing))]
    pub data: Vec<bytes::Bytes>,
}

impl From<EvaluateOutput> for EvaluateOutputBinary {
    fn from(value: EvaluateOutput) -> Self {
        Self {
            name: value.name,
            data: value.data.into_iter().map(|x| x.serialize()).collect(),
        }
    }
}

pub trait WeightsLoader: Send + Sync {
    fn byte_size(&self) -> u64;

    fn read(&mut self) -> BoxFuture<'_, bytes::Bytes>;
}

pub trait WeightsProvider: Send + Sync {
    fn provide_all<'a>(
        &'a mut self,
        data: &'a [(impl AsRef<str> + Send + Sync + 'a, bytes::Bytes)],
    ) -> BoxFuture<'a, ()>;

    fn provide<'a>(&'a mut self, key: &'a str, data: bytes::Bytes) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            self.provide_all(&[(key, data)]).await;
        })
    }
}

pub trait DataLoaderBinary: Send + Sync {
    fn total_byte_size(&self) -> u64;

    /// After this has been called, the data points will be returned in a random order from future
    /// reads.
    fn shuffle(&self) -> BoxFuture<'_, ()> {
        self.shuffle_in_group(&[])
    }

    /// After this has been called, the data points will be returned in a random order from future
    /// reads. The data loaders in *others* will be shuffled in the same order.
    fn shuffle_in_group<'a>(&'a self, others: &'a [&'a Self]) -> BoxFuture<'a, ()>;

    fn size(&self) -> u32;

    fn position(&self) -> u32;

    /// After called, future reads will read from this position instead.
    fn set_position(&mut self, position: u32);

    /// Returns the number of remaining data points, i.e self.size() - self.position().
    fn remaining(&self) -> u32 {
        DataLoaderBinary::size(self) - DataLoaderBinary::position(self)
    }

    /// Returns true if there are data left to fetch, i.e self.remaining() >= amount
    fn has_next(&self, amount: u32) -> bool {
        DataLoaderBinary::remaining(self) >= amount
    }

    /// Fetches data points and advance the position by *amount*. If self.remaining() is less
    /// than *amount*, self.remaining() data points are fetched instead.
    fn next(&mut self, amount: u32) -> BoxFuture<'_, Vec<bytes::Bytes>>;
}

pub trait DataLoader: Send + Sync {
    fn total_byte_size(&self) -> u64;

    /// After this has been called, the data points will be returned in a random order from future
    /// reads.
    fn shuffle(&self) -> BoxFuture<'_, ()> {
        self.shuffle_in_group(&[])
    }

    /// After this has been called, the data points will be returned in a random order from future
    /// reads. The data loaders in *others* will be shuffled in the same order.
    fn shuffle_in_group<'a>(&'a self, others: &'a [&'a Self]) -> BoxFuture<'a, ()>;

    fn size(&self) -> u32;

    fn position(&self) -> u32;

    /// After called, future reads will read from this position instead.
    fn set_position(&mut self, position: u32);

    /// Returns the number of remaining data points, i.e self.size() - self.position().
    fn remaining(&self) -> u32 {
        DataLoader::size(self) - DataLoader::position(self)
    }

    /// Returns true if there are data left to fetch, i.e self.remaining() >= amount
    fn has_next(&self, amount: u32) -> bool {
        DataLoader::remaining(self) >= amount
    }

    /// Fetches data and advances the position by *amount*. If self.remaining() is less than
    /// *amount*, self.remaining() data points are fetched instead.
    fn next(&mut self, amount: u32) -> BoxFuture<'_, Vec<OwnedDecthingsTensor>>;
}

impl<T: DataLoaderBinary + Send> DataLoader for T {
    fn total_byte_size(&self) -> u64 {
        DataLoaderBinary::total_byte_size(self)
    }

    fn shuffle_in_group<'a>(&'a self, others: &'a [&'a Self]) -> BoxFuture<'a, ()> {
        DataLoaderBinary::shuffle_in_group(self, others)
    }

    fn size(&self) -> u32 {
        DataLoaderBinary::size(self)
    }

    fn position(&self) -> u32 {
        DataLoaderBinary::position(self)
    }

    fn set_position(&mut self, position: u32) {
        DataLoaderBinary::set_position(self, position)
    }

    fn remaining(&self) -> u32 {
        DataLoaderBinary::remaining(self)
    }

    fn has_next(&self, amount: u32) -> bool {
        DataLoaderBinary::has_next(self, amount)
    }

    fn next(&mut self, amount: u32) -> BoxFuture<'_, Vec<OwnedDecthingsTensor>> {
        Box::pin(async move {
            DataLoaderBinary::next(self, amount)
                .await
                .into_iter()
                .map(|x| OwnedDecthingsTensor::from_bytes(x).unwrap())
                .collect()
        })
    }
}

#[derive(Clone, Debug)]
pub struct MetricBinary<S: AsRef<str>> {
    pub name: S,
    pub data: bytes::Bytes,
}

pub trait TrainTrackerBinary: Send + Sync {
    fn wait_for_cancelled(&self) -> BoxFuture<'_, ()>;

    fn progress(&self, progress: f32) -> BoxFuture<'_, ()>;

    fn metrics<'a>(
        &'a self,
        metrics: &'a [MetricBinary<impl AsRef<str> + Sync + 'a>],
    ) -> BoxFuture<'a, ()>;
}

#[derive(Clone, Debug)]
pub struct Metric<'a, S: AsRef<str>> {
    pub name: S,
    pub data: DecthingsTensor<'a>,
}

pub trait TrainTracker: Send + Sync {
    fn wait_for_cancelled(&self) -> BoxFuture<'_, ()>;

    fn progress(&self, progress: f32) -> BoxFuture<'_, ()>;

    fn metrics<'a, 'b>(
        &'a self,
        metrics: &'a [Metric<'b, impl AsRef<str> + Sync + 'a>],
    ) -> BoxFuture<'a, ()>
    where
        'b: 'a;
}

impl<T: TrainTrackerBinary> TrainTracker for T {
    fn wait_for_cancelled(&self) -> BoxFuture<'_, ()> {
        TrainTrackerBinary::wait_for_cancelled(self)
    }

    fn progress(&self, progress: f32) -> BoxFuture<'_, ()> {
        TrainTrackerBinary::progress(self, progress)
    }

    fn metrics<'a, 'b>(
        &'a self,
        metrics: &'a [Metric<'b, impl AsRef<str> + Sync + 'a>],
    ) -> BoxFuture<'a, ()>
    where
        'b: 'a,
    {
        Box::pin(async move {
            TrainTrackerBinary::metrics(
                self,
                &metrics
                    .iter()
                    .map(|x| MetricBinary {
                        name: x.name.as_ref(),
                        data: x.data.serialize().into(),
                    })
                    .collect::<Vec<_>>(),
            )
            .await;
        })
    }
}

#[derive(Clone, Debug)]
pub struct ExpectedOutputType {
    pub required: bool,
    pub rules: decthings_api::tensor::DecthingsTensorRules,
}

#[derive(Clone, Debug)]
pub struct EvaluateOptions<D> {
    pub params: HashMap<String, D>,
    pub expected_output_types: HashMap<String, ExpectedOutputType>,
}

#[derive(Clone, Debug)]
pub struct TrainOptions<D, T> {
    pub params: HashMap<String, D>,
    pub tracker: T,
}

#[derive(Clone, Debug)]
pub struct GetWeightsOptions<WP: WeightsProvider> {
    pub weights_provider: WP,
}

pub trait InstantiatedBinary: Send + Sync {
    fn evaluate<'a>(
        &'a self,
        options: EvaluateOptions<impl DataLoaderBinary + 'a>,
    ) -> BoxFuture<'a, Vec<EvaluateOutputBinary>> {
        let _ = options;
        panic!("Evaluate was called but was not implemented.");
    }

    fn train<'a>(
        &'a self,
        options: TrainOptions<impl DataLoaderBinary + 'a, impl TrainTrackerBinary + 'a>,
    ) -> BoxFuture<'a, ()> {
        let _ = options;
        panic!("Train was called but was not implemented.");
    }

    fn get_weights<'a>(
        &'a self,
        options: GetWeightsOptions<impl WeightsProvider + 'a>,
    ) -> BoxFuture<'a, ()> {
        let _ = options;
        panic!("GetWeights was called but was not implemented.");
    }
}

#[derive(Clone, Debug)]
pub struct OtherModelWithWeights<WL: WeightsLoader> {
    pub mount_path: String,
    pub weights: HashMap<String, WL>,
}

#[derive(Clone, Debug)]
pub struct InitializeWeightsOptions<D, WP: WeightsProvider, WL: WeightsLoader> {
    pub params: HashMap<String, D>,
    pub weights_provider: WP,
    pub other_models: HashMap<String, OtherModelWithWeights<WL>>,
}

#[derive(Clone, Debug)]
pub struct OtherModel {
    pub mount_path: String,
}

#[derive(Clone, Debug)]
pub struct InstantiateModelOptions<WL: WeightsLoader> {
    pub weights: HashMap<String, WL>,
    pub other_models: HashMap<String, OtherModel>,
}

pub trait ModelBinary: Send + Sync {
    type Instantiated: InstantiatedBinary;

    fn initialize_weights<'a>(
        options: InitializeWeightsOptions<
            impl DataLoaderBinary + 'a,
            impl WeightsProvider + 'a,
            impl WeightsLoader + 'a,
        >,
    ) -> BoxFuture<'a, ()> {
        let _ = options;
        panic!("InitializeWeights was called but was not implemented.");
    }

    fn instantiate_model<'a>(
        options: InstantiateModelOptions<impl WeightsLoader + 'a>,
    ) -> BoxFuture<'a, Self::Instantiated> {
        let _ = options;
        panic!("InstantiateModel was called but was not implemented.");
    }
}

pub trait Instantiated: Send + Sync {
    fn evaluate<'a>(
        &'a self,
        options: EvaluateOptions<impl DataLoader + 'a>,
    ) -> BoxFuture<'a, Vec<EvaluateOutput>> {
        let _ = options;
        panic!("Evaluate was called but was not implemented.");
    }

    fn train<'a>(
        &'a self,
        options: TrainOptions<impl DataLoader + 'a, impl TrainTracker + 'a>,
    ) -> BoxFuture<'a, ()> {
        let _ = options;
        panic!("Train was called but was not implemented.");
    }

    fn get_weights<'a>(
        &'a self,
        options: GetWeightsOptions<impl WeightsProvider + 'a>,
    ) -> BoxFuture<'a, ()> {
        let _ = options;
        panic!("GetWeights was called but was not implemented.");
    }
}

impl<T: Instantiated + Sync> InstantiatedBinary for T {
    fn evaluate<'a>(
        &'a self,
        options: EvaluateOptions<impl DataLoaderBinary + 'a>,
    ) -> BoxFuture<'a, Vec<EvaluateOutputBinary>> {
        Box::pin(async move {
            let res = T::evaluate(self, options).await;
            res.into_iter().map(|x| x.into()).collect()
        })
    }

    fn train<'a>(
        &'a self,
        options: TrainOptions<impl DataLoaderBinary + 'a, impl TrainTrackerBinary + 'a>,
    ) -> BoxFuture<'a, ()> {
        T::train(self, options)
    }

    fn get_weights<'a>(
        &'a self,
        options: GetWeightsOptions<impl WeightsProvider + 'a>,
    ) -> BoxFuture<'a, ()> {
        T::get_weights(self, options)
    }
}

pub trait Model: Send + Sync {
    type Instantiated: Instantiated;

    fn initialize_weights<'a>(
        options: InitializeWeightsOptions<
            impl DataLoader + 'a,
            impl WeightsProvider + 'a,
            impl WeightsLoader + 'a,
        >,
    ) -> BoxFuture<'a, ()> {
        let _ = options;
        panic!("InitializeWeights was called but was not implemented.");
    }

    fn instantiate_model<'a>(
        options: InstantiateModelOptions<impl WeightsLoader + 'a>,
    ) -> BoxFuture<'a, Self::Instantiated> {
        let _ = options;
        panic!("InstantiateModel was called but was not implemented.");
    }
}

impl<T: Model> ModelBinary for T
where
    T::Instantiated: Sync,
{
    type Instantiated = T::Instantiated;

    fn initialize_weights<'a>(
        options: InitializeWeightsOptions<
            impl DataLoaderBinary + 'a,
            impl WeightsProvider + 'a,
            impl WeightsLoader + 'a,
        >,
    ) -> BoxFuture<'a, ()> {
        T::initialize_weights(options)
    }

    fn instantiate_model<'a>(
        options: InstantiateModelOptions<impl WeightsLoader + 'a>,
    ) -> BoxFuture<'a, Self::Instantiated> {
        T::instantiate_model(options)
    }
}
