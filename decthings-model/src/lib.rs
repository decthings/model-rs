#[cfg(target_family = "wasm")]
mod wasm;

#[cfg(target_family = "wasm")]
pub use wasm::*;

#[cfg(target_family = "unix")]
mod unix;

#[cfg(target_family = "unix")]
pub use unix::*;

mod parameter;
mod trait_def;

pub use parameter::*;
pub use trait_def::*;
