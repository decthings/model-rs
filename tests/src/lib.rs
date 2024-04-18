use std::{future::Future, pin::Pin};

struct Instantiated;

impl decthings_model::Instantiated for Instantiated {
    fn train<'a>(
        &'a self,
        _options: decthings_model::TrainOptions<
            impl decthings_model::DataLoader + 'a,
            impl decthings_model::TrainTracker + 'a,
        >,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        todo!()
    }

    fn evaluate<'a>(
        &'a self,
        _options: decthings_model::EvaluateOptions<impl decthings_model::DataLoader + 'a>,
    ) -> Pin<Box<dyn Future<Output = Vec<decthings_model::Parameter<'a>>> + Send + 'a>> {
        todo!()
    }

    fn get_model_state<'a>(
        &'a self,
        _options: decthings_model::GetModelStateOptions<impl decthings_model::StateProvider + 'a>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        todo!()
    }
}

struct Model;

impl decthings_model::Model for Model {
    type Instantiated = Instantiated;

    fn create_model_state<'a>(
        &'a self,
        _options: decthings_model::CreateModelStateOptions<
            impl decthings_model::DataLoader + 'a,
            impl decthings_model::StateProvider + 'a,
            impl decthings_model::StateLoader + 'a,
        >,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        todo!()
    }

    fn instantiate_model<'a>(
        &'a self,
        _options: decthings_model::InstantiateModelOptions<impl decthings_model::StateLoader + 'a>,
    ) -> Pin<Box<dyn Future<Output = Self::Instantiated> + Send + 'a>> {
        todo!()
    }
}

#[cfg(target_family = "wasm")]
#[decthings_model::decthings_initialize]
fn create_model() -> Model {
    Model
}
