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

    let struct_name = item_struct.ident;
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
    let full_struct_name = quote!(#struct_name<#(#generics),*>);

    let mut attributes = Vec::new();
    for field in fields.named.iter() {
        let name = field.ident.as_ref().unwrap();
        let ty = &field.ty;
        //TODO: use this when MSRV gets to 1.77
        // `std::mem::offset_of!(#full_struct_name, #name)
        attributes.push(quote! {
            (stringify!(#name), blade_graphics::VertexAttribute {
                offset: unsafe {
                    (&(*base_ptr).#name as *const _ as *const u8).offset_from(base_ptr as *const u8) as u32
                },
                format: <#ty as blade_graphics::derive::HasVertexAttribute>::FORMAT,
            })
        });
    }

    Ok(quote! {
        impl<#(#generics),*> blade_graphics::Vertex for #full_struct_name {
            fn layout() -> blade_graphics::VertexLayout {
                let uninit = <core::mem::MaybeUninit<Self>>::uninit();
                let base_ptr = uninit.as_ptr();
                blade_graphics::VertexLayout {
                    attributes: vec![#(#attributes),*],
                    stride: core::mem::size_of::<Self>() as u32,
                }
            }
        }
    })
}
