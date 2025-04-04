use std::{future::Future, pin::Pin};

// This file checks if a wasm model can compile successfully. To test, run
// cd wasm-compile-test && cargo component check --target wasm32-wasi

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
    ) -> Pin<Box<dyn Future<Output = Vec<decthings_model::EvaluateOutput>> + Send + 'a>> {
        todo!()
    }

    fn get_weights<'a>(
        &'a self,
        _options: decthings_model::GetWeightsOptions<impl decthings_model::WeightsProvider + 'a>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        todo!()
    }
}

struct Model;

impl decthings_model::Model for Model {
    type Instantiated = Instantiated;

    fn initialize_weights<'a>(
        _options: decthings_model::InitializeWeightsOptions<
            impl decthings_model::DataLoader + 'a,
            impl decthings_model::WeightsProvider + 'a,
            impl decthings_model::WeightsLoader + 'a,
        >,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        todo!()
    }

    fn instantiate_model<'a>(
        _options: decthings_model::InstantiateModelOptions<impl decthings_model::WeightsLoader + 'a>,
    ) -> Pin<Box<dyn Future<Output = Self::Instantiated> + Send + 'a>> {
        todo!()
    }
}

#[cfg(target_family = "wasm")]
mod bindings;

#[cfg(target_family = "wasm")]
decthings_model::wasm_bindings::export_decthings_model!(Model with_types_in bindings);
