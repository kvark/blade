use proc_macro::TokenStream;
use quote::quote;

enum ResourceType {
    TextureView,
    Sampler,
    Buffer,
}

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
        let res_type = match field.ty {
            syn::Type::Path(ref typ) => {
                let idents = typ
                    .path
                    .segments
                    .iter()
                    .map(|segment| segment.ident.to_string())
                    .collect::<Vec<_>>();
                if idents.len() == 2 && idents[0] == "blade" {
                    Some(match idents[1].as_str() {
                        "BufferPiece" => ResourceType::Buffer,
                        "TextureView" => ResourceType::TextureView,
                        "Sampler" => ResourceType::Sampler,
                        _ => {
                            return Err(syn::Error::new(
                                typ.path.segments[1].ident.span(),
                                "Unknown Blade resource type",
                            ))
                        }
                    })
                } else {
                    None
                }
            }
            //TODO: arrays of plain data, textures, etc
            //syn::Type::Array(ref typ) => {}
            //syn::Type::Slice(ref typ) => {}
            _ => None,
        };

        let ty = &field.ty;
        let (setter, value) = match res_type {
            Some(ResourceType::Buffer) => (
                "set_buffer",
                quote!(blade::ShaderBinding::Buffer {
                    type_name: "", //TODO
                    access: blade::StorageAccess::all(),
                }),
            ),
            Some(ResourceType::TextureView) => (
                "set_texture",
                quote!(blade::ShaderBinding::Texture {
                    dimension: blade::TextureViewDimension::D2,
                }),
            ),
            Some(ResourceType::Sampler) => (
                "set_sampler",
                quote!(blade::ShaderBinding::Sampler { comparison: false }),
            ),
            None => (
                "set_plain",
                quote!(
                    blade::ShaderBinding::Plain {
                        ty: <#ty as blade::AsPlain>::TYPE,
                        container: <#ty as blade::AsPlain>::CONTAINER,
                    }
                ),
            ),
        };

        let name = field.ident.as_ref().unwrap();
        let setter_ident = syn::Ident::new(setter, proc_macro2::Span::call_site());
        let index = index_usize as u32;
        bindings.push(quote! {
            (stringify!(#name), #value)
        });
        assignments.push(quote! {
            encoder.#setter_ident(#index, self.#name);
        });
    }

    let impl_layout = quote! {
        fn layout() -> blade::ShaderDataLayout {
            blade::ShaderDataLayout {
                bindings: vec![#(#bindings),*],
            }
        }
        fn fill<E: blade::ShaderDataEncoder>(&self, mut encoder: E) {
            #(#assignments)*
        }
    };

    let struct_name = item_struct.ident;
    let output = quote! {
        impl blade::ShaderData for #struct_name {
            #impl_layout
        }
    };

    Ok(output)
}

#[proc_macro_derive(ShaderData)]
pub fn shader_data_derive(input: TokenStream) -> TokenStream {
    let stream = match impl_shader_data(input) {
        Ok(tokens) => tokens,
        Err(err) => err.into_compile_error(),
    };
    stream.into()
}
