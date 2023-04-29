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

    let mut expr_alignment = quote!(0);
    let mut expr_size = quote!(0);
    let mut st_write = Vec::new();
    let mut init_read = Vec::new();
    for field in fields.named.iter() {
        let name = field.ident.as_ref().unwrap();
        let ty = &field.ty;
        let align = quote! { <#ty as blade_asset::Flat>::ALIGNMENT };
        expr_alignment = quote! {
            [#expr_alignment, #align][(#expr_alignment < #align) as usize]
        };
        expr_size = quote! {
            blade_asset::round_up(#expr_size, #align) + self.#name.size()
        };
        st_write.push(quote! {
            ptr = ptr.add(ptr.align_offset(#align));
            self.#name.write(ptr);
            ptr = ptr.add(self.#name.size());
        });
        init_read.push(quote! {
            #name: {
                ptr = ptr.add(ptr.align_offset(#align));
                let value = <#ty as blade_asset::Flat>::read(ptr);
                ptr = ptr.add(blade_asset::Flat::size(&value));
                value
            },
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
        impl<#(#generics),*> blade_asset::Flat for #struct_name<#(#generics),*> {
            const ALIGNMENT: usize = #expr_alignment;
            //Note: this could be improved if we see all fields being `FIXED_SIZE`
            const FIXED_SIZE: Option<std::num::NonZeroUsize> = None;
            fn size(&self) -> usize {
                #expr_size
            }
            unsafe fn write(&self, mut ptr: *mut u8) {
                #(#st_write)*
            }
            unsafe fn read(mut ptr: *const u8) -> Self {
                Self {
                    #(#init_read)*
                }
            }
        }
    })
}
