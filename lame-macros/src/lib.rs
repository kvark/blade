use proc_macro::TokenStream;
use quote::quote;

/*
    let ident = &input.item_struct.ident;
    let layout_ident = layout_type_name(input);
    let instance_ident = instance_type_name(input);

    quote::quote! {
        impl ::sierra::Descriptors for #ident {
            type Layout = #layout_ident;
            type Instance = #instance_ident;

            fn layout(device: &sierra::Device) -> ::std::result::Result<Self::Layout, ::sierra::OutOfMemory> {
                #layout_ident::new(device)
            }
        }
    }
*/

#[proc_macro_derive(ShaderData)]
pub fn shader_data_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let impl_layout = quote!{
        fn layout() -> crate::ShaderDataLayout {
            crate::ShaderDataLayout {
                entries: vec![],
            }
        }
    };

    let struct_name = ast.ident;
    let output = quote!{
        impl crate::ShaderData for #struct_name {
            #impl_layout
        }
    };

    output.into()
}
