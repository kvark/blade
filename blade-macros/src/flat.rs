use proc_macro::TokenStream;
use quote::quote;

pub fn generate(input_stream: TokenStream) -> syn::Result<proc_macro2::TokenStream> {
    let item_struct = syn::parse::<syn::ItemStruct>(input_stream)?;

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

    Ok(match item_struct.fields {
        syn::Fields::Unnamed(_) => {
            let is_transparent = item_struct.attrs.iter().any(|attr| {
                if !attr.path().is_ident("repr") {
                    return false;
                }
                let value = attr.parse_args::<syn::Ident>().unwrap();
                value == "transparent"
            });
            if !is_transparent {
                return Err(syn::Error::new(
                    item_struct.struct_token.span,
                    "Only `repr(transparent)` wrappers are supported",
                ));
            }

            quote! {
                impl<#(#generics),*> blade_asset::Flat for #struct_name<#(#generics),*> {
                    const ALIGNMENT: usize = std::mem::size_of::<Self>();
                    const FIXED_SIZE: Option<std::num::NonZeroUsize> = std::num::NonZeroUsize::new(std::mem::size_of::<Self>());
                    unsafe fn write(&self, mut ptr: *mut u8) {
                        std::ptr::write(ptr as *mut Self, *self);
                    }
                    unsafe fn read(mut ptr: *const u8) -> Self {
                        std::ptr::read(ptr as *const Self)
                    }
                }
            }
        }
        syn::Fields::Named(ref fields) => {
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
                        ptr = ptr.add(value.size());
                        value
                    },
                });
            }

            quote! {
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
            }
        }
        _ => {
            return Err(syn::Error::new(
                item_struct.struct_token.span,
                "Structure fields must be named",
            ))
        }
    })
}
