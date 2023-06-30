use proc_macro::TokenStream;
use quote::quote;

pub fn generate(input_stream: TokenStream) -> syn::Result<proc_macro2::TokenStream> {
    let item_enum = syn::parse::<syn::ItemEnum>(input_stream)?;
    let enum_name = item_enum.ident;
    Ok(quote! {
        impl Into<u32> for #enum_name {
            fn into(self) -> u32 {
                self as u32
            }
        }
    })
}
