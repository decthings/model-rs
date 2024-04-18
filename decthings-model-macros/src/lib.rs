use proc_macro::TokenStream;
use syn::spanned::Spanned;

struct ParsedFn {
    outer_attrs: Vec<syn::Attribute>,
    vis: syn::Visibility,
    sig: syn::Signature,
    content: syn::ExprBlock,
}

impl syn::parse::Parse for ParsedFn {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let outer_attrs = input.call(syn::Attribute::parse_outer)?;
        let vis: syn::Visibility = input.parse()?;
        let sig: syn::Signature = input.parse()?;
        let content: syn::ExprBlock = input.parse()?;

        if !sig.inputs.is_empty() {
            return Err(syn::Error::new(
                sig.inputs.span(),
                "Expected zero arguments",
            ));
        }

        let ret_type = &match sig.output {
            syn::ReturnType::Default => Box::new(syn::parse_str::<syn::Type>("()").unwrap()),
            syn::ReturnType::Type(_, ref t) => t.to_owned(),
        };
        if let syn::Type::ImplTrait(_) = &**ret_type {
            return Err(syn::Error::new(
                ret_type.span(),
                "impl Trait is not supported by decthings_initialize. Specify the type instead.",
            ));
        }

        Ok(Self {
            outer_attrs,
            vis,
            sig,
            content,
        })
    }
}

#[proc_macro_attribute]
pub fn decthings_initialize(_args: TokenStream, item: TokenStream) -> TokenStream {
    let ParsedFn {
        outer_attrs,
        vis,
        sig,
        content,
    } = syn::parse_macro_input!(item as ParsedFn);
    let fn_name = &sig.ident;
    let ret_type = &match sig.output {
        syn::ReturnType::Default => Box::new(syn::parse_str::<syn::Type>("()").unwrap()),
        syn::ReturnType::Type(_, ref t) => t.to_owned(),
    };

    quote::quote! {
        #(#outer_attrs)*
        #vis
        #sig
        #content

        ::decthings_model::lazy_static! {
            static ref __DECTHINGS_MODEL: #ret_type = #fn_name();
            static ref __DECTHINGS_INSTANTIATED_MODELS:
                ::std::sync::Mutex<
                    ::std::collections::HashMap<
                        u32,
                        <#ret_type as ::decthings_model::ModelBinary>::Instantiated
                    >
                > = ::std::sync::Mutex::new(::std::collections::HashMap::new());
        }

        #[no_mangle]
        pub extern "C" fn dt_initialize() {
            let _= &*__DECTHINGS_MODEL;
        }

        #[no_mangle]
        pub unsafe extern "C" fn dt_create_model_state(params_ptr: u64, other_models_ptr: u64) {
            use ::decthings_model::ModelBinary;

            let params = ::decthings_model::wasm_host_functions::read_params(params_ptr);

            let fut = ::decthings_model::ModelBinary::create_model_state(
                &*__DECTHINGS_MODEL,
                ::decthings_model::CreateModelStateOptions {
                    params,
                    state_provider: ::decthings_model::wasm_host_functions::get_state_provider(),
                    other_models: ::decthings_model::wasm_host_functions::read_other_models_with_state(other_models_ptr),
                },
            );

            ::decthings_model::pollster::block_on(fut);

            ::decthings_model::wasm_host_functions::dt_callback(0);
        }

        #[no_mangle]
        pub unsafe extern "C" fn dt_instantiate_model(id: u32, state_ptr: u64, other_models_ptr: u64) {
            let state = ::decthings_model::wasm_host_functions::read_params(state_ptr);

            let fut = ::decthings_model::ModelBinary::instantiate_model(
                &*__DECTHINGS_MODEL,
                ::decthings_model::InstantiateModelOptions {
                    state,
                    other_models: ::decthings_model::wasm_host_functions::read_other_models(other_models_ptr),
                }
            );

            let instantiated = ::decthings_model::pollster::block_on(fut);
            __DECTHINGS_INSTANTIATED_MODELS.lock().unwrap().insert(id, instantiated);

            ::decthings_model::wasm_host_functions::dt_callback(0);
        }

        #[no_mangle]
        pub unsafe extern "C" fn dt_evaluate(instantiated_model_id: u32, params_ptr: u64) {
            let params = ::decthings_model::wasm_host_functions::read_params(params_ptr);

            let locked = __DECTHINGS_INSTANTIATED_MODELS.lock().unwrap();
            let instantiated = locked.get(&instantiated_model_id).unwrap();

            let fut = ::decthings_model::InstantiatedBinary::evaluate(
                instantiated,
                ::decthings_model::EvaluateOptions {
                    params,
                },
            );
            let output = ::decthings_model::pollster::block_on(fut);

            ::decthings_model::wasm_host_functions::evaluate_callback(Ok(output));
        }

        #[no_mangle]
        pub unsafe extern "C" fn dt_train(instantiated_model_id: u32, params_ptr: u64) {
            let params = ::decthings_model::wasm_host_functions::read_params(params_ptr);

            let locked = __DECTHINGS_INSTANTIATED_MODELS.lock().unwrap();
            let instantiated = locked.get(&instantiated_model_id).unwrap();

            let fut = ::decthings_model::InstantiatedBinary::train(
                instantiated,
                ::decthings_model::TrainOptions {
                    params,
                    tracker: ::decthings_model::wasm_host_functions::get_train_tracker(),
                },
            );
            ::decthings_model::pollster::block_on(fut);

            ::decthings_model::wasm_host_functions::dt_callback(0);
        }

        #[no_mangle]
        pub unsafe extern "C" fn dt_get_model_state(instantiated_model_id: u32) {
            let locked = __DECTHINGS_INSTANTIATED_MODELS.lock().unwrap();
            let instantiated = locked.get(&instantiated_model_id).unwrap();

            let fut = ::decthings_model::InstantiatedBinary::get_model_state(
                instantiated,
                ::decthings_model::GetModelStateOptions {
                    state_provider: ::decthings_model::wasm_host_functions::get_state_provider(),
                },
            );
            ::decthings_model::pollster::block_on(fut);

            ::decthings_model::wasm_host_functions::dt_callback(0);
        }
    }
    .into()
}
