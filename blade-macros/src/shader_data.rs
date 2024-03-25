use proc_macro::TokenStream;
use quote::quote;

pub fn generate(input_stream: TokenStream) -> syn::Result<proc_macro2::TokenStream> {
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
            (stringify!(#name), <#ty as blade_graphics::derive::HasShaderBinding>::TYPE)
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
        impl<#(#generics),*> blade_graphics::ShaderData for #struct_name<#(#generics),*> {
            fn layout() -> blade_graphics::ShaderDataLayout {
                blade_graphics::ShaderDataLayout {
                    bindings: vec![#(#bindings),*],
                }
            }
            fn fill(&self, mut ctx: blade_graphics::PipelineContext) {
                use blade_graphics::ShaderBindable as _;
                #(#assignments)*
            }
        }
    })
}
