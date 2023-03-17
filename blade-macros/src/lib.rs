use proc_macro::TokenStream;
use quote::quote;

fn impl_shader_data(input_stream: TokenStream) -> syn::Result<proc_macro2::TokenStream> {
    let item_struct = syn::parse::<syn::ItemStruct>(input_stream)?;
    let fields = match item_struct.fields {
        syn::Fields::Named(ref fields) => fields,
        _ => {
            return Err(syn::Error::new(
                item_struct.struct_token.span,
                "Structure fields must be named",
            ))
        }
    };

    let mut bindings = Vec::new();
    let mut assignments = Vec::new();
    for (index_usize, field) in fields.named.iter().enumerate() {
        let index = index_usize as u32;
        let name = field.ident.as_ref().unwrap();
        let ty = &field.ty;
        bindings.push(quote! {
            (stringify!(#name), <#ty as blade::HasShaderBinding>::TYPE)
        });
        assignments.push(quote! {
            self.#name.bind_to(&mut ctx, #index);
        });
    }

    let mut generics = Vec::new();
    for param in item_struct.generics.params {
        match param {
            syn::GenericParam::Lifetime(lt) => {
                generics.push(lt.lifetime);
            }
            syn::GenericParam::Type(_) | syn::GenericParam::Const(_) => {
                return Err(syn::Error::new(
                    item_struct.struct_token.span,
                    "Unsupported generic parameters",
                ))
            }
        }
    }

    let struct_name = item_struct.ident;
    Ok(quote! {
        impl<#(#generics),*> blade::ShaderData for #struct_name<#(#generics),*> {
            fn layout() -> blade::ShaderDataLayout {
                blade::ShaderDataLayout {
                    bindings: vec![#(#bindings),*],
                }
            }
            fn fill(&self, mut ctx: blade::PipelineContext) {
                use blade::ShaderBindable as _;
                #(#assignments)*
            }
        }
    })
}

#[proc_macro_derive(ShaderData)]
pub fn shader_data_derive(input: TokenStream) -> TokenStream {
    let stream = match impl_shader_data(input) {
        Ok(tokens) => tokens,
        Err(err) => err.into_compile_error(),
    };
    stream.into()
}
