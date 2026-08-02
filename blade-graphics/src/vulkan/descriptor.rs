use ash::vk;

//TODO: replace by an abstraction in `gpu-descriptor`
// https://github.com/zakarumych/gpu-descriptor/issues/42
const COUNT_BASE: u32 = 16;
/// Budget for inline uniform block bytes per descriptor set.
/// The hardware max (e.g. 4 MiB on RADV) is far larger than actual
/// usage (typically 32–256 bytes of push constants per set).
/// Using the hardware max as the multiplier causes pool creation to
/// request more memory than the device has (e.g. 4096 sets × 4 MiB = 16 GiB).
const IUB_BYTES_PER_SET: u32 = 4096;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct DescriptorCounts {
    pub storage_buffers: u32,
    pub sampled_images: u32,
    pub samplers: u32,
    pub storage_images: u32,
    pub inline_uniform_bytes: u32,
    pub inline_uniform_bindings: u32,
    pub uniform_buffers: u32,
    pub acceleration_structures: u32,
}

impl DescriptorCounts {
    pub fn add(&mut self, ty: vk::DescriptorType, count: u32) {
        match ty {
            vk::DescriptorType::STORAGE_BUFFER => self.storage_buffers += count,
            vk::DescriptorType::SAMPLED_IMAGE => self.sampled_images += count,
            vk::DescriptorType::SAMPLER => self.samplers += count,
            vk::DescriptorType::STORAGE_IMAGE => self.storage_images += count,
            vk::DescriptorType::INLINE_UNIFORM_BLOCK_EXT => {
                self.inline_uniform_bytes += count;
                self.inline_uniform_bindings += 1;
            }
            vk::DescriptorType::UNIFORM_BUFFER => self.uniform_buffers += count,
            vk::DescriptorType::ACCELERATION_STRUCTURE_KHR => {
                self.acceleration_structures += count;
            }
            _ => unreachable!("unsupported descriptor type {ty:?}"),
        }
    }

    fn max(self, other: Self) -> Self {
        Self {
            storage_buffers: self.storage_buffers.max(other.storage_buffers),
            sampled_images: self.sampled_images.max(other.sampled_images),
            samplers: self.samplers.max(other.samplers),
            storage_images: self.storage_images.max(other.storage_images),
            inline_uniform_bytes: self.inline_uniform_bytes.max(other.inline_uniform_bytes),
            inline_uniform_bindings: self
                .inline_uniform_bindings
                .max(other.inline_uniform_bindings),
            uniform_buffers: self.uniform_buffers.max(other.uniform_buffers),
            acceleration_structures: self
                .acceleration_structures
                .max(other.acceleration_structures),
        }
    }

    fn supports(self, required: Self) -> bool {
        self.max(required) == self
    }
}

#[derive(Debug)]
struct DescriptorSubPool {
    raw: vk::DescriptorPool,
    max_sets: u32,
    allocated_sets: u32,
    per_set: DescriptorCounts,
}

#[derive(Debug)]
pub struct DescriptorPool {
    sub_pools: Vec<DescriptorSubPool>,
}

impl super::Device {
    fn create_descriptor_sub_pool(
        &self,
        max_sets: u32,
        required_per_set: DescriptorCounts,
    ) -> DescriptorSubPool {
        log::info!("Creating a descriptor pool for at most {} sets", max_sets);
        let baseline = DescriptorCounts {
            storage_buffers: 1,
            sampled_images: 2,
            samplers: 1,
            storage_images: 1,
            inline_uniform_bytes: if self.max_inline_uniform_block_size > 0 {
                IUB_BYTES_PER_SET
            } else {
                0
            },
            inline_uniform_bindings: u32::from(self.max_inline_uniform_block_size > 0),
            uniform_buffers: 1,
            acceleration_structures: u32::from(self.ray_tracing.is_some()),
        };
        let per_set = baseline.max(required_per_set);
        let pool_count = |count: u32| {
            count
                .checked_mul(max_sets)
                .expect("descriptor pool count overflow")
        };
        let mut descriptor_sizes = vec![
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: pool_count(per_set.storage_buffers),
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: pool_count(per_set.sampled_images),
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: pool_count(per_set.samplers),
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: pool_count(per_set.storage_images),
            },
        ];
        if self.max_inline_uniform_block_size > 0 {
            descriptor_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::INLINE_UNIFORM_BLOCK_EXT,
                descriptor_count: pool_count(per_set.inline_uniform_bytes),
            });
        }
        // Always include UBO type: needed as fallback when bindings exceed
        // the inline uniform block size limit, or when IUBs aren't supported.
        descriptor_sizes.push(vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: pool_count(per_set.uniform_buffers),
        });
        if self.ray_tracing.is_some() {
            descriptor_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count: pool_count(per_set.acceleration_structures),
            });
        }

        let mut inline_uniform_block_info = vk::DescriptorPoolInlineUniformBlockCreateInfoEXT {
            max_inline_uniform_block_bindings: pool_count(per_set.inline_uniform_bindings),
            ..Default::default()
        };

        let mut descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .flags(self.workarounds.extra_descriptor_pool_create_flags)
            .pool_sizes(&descriptor_sizes);
        if self.max_inline_uniform_block_size > 0 {
            descriptor_pool_info = descriptor_pool_info.push_next(&mut inline_uniform_block_info);
        }

        let raw = unsafe {
            self.core
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap()
        };
        DescriptorSubPool {
            raw,
            max_sets,
            allocated_sets: 0,
            per_set,
        }
    }

    pub(super) fn create_descriptor_pool(&self) -> DescriptorPool {
        let sub_pool = self.create_descriptor_sub_pool(COUNT_BASE, DescriptorCounts::default());
        DescriptorPool {
            sub_pools: vec![sub_pool],
        }
    }

    pub(super) fn destroy_descriptor_pool(&self, pool: &mut DescriptorPool) {
        for sub_pool in pool.sub_pools.drain(..) {
            unsafe { self.core.destroy_descriptor_pool(sub_pool.raw, None) };
        }
    }

    pub(super) fn allocate_descriptor_set(
        &self,
        pool: &mut DescriptorPool,
        layout: &super::DescriptorSetLayout,
    ) -> vk::DescriptorSet {
        let descriptor_set_layouts = [layout.raw];

        loop {
            let needs_larger_pool = {
                let sub_pool = &pool.sub_pools[0];
                sub_pool.allocated_sets == sub_pool.max_sets
                    || !sub_pool.per_set.supports(layout.descriptor_counts)
            };
            if needs_larger_pool {
                let sub_pool = &pool.sub_pools[0];
                let max_sets = if sub_pool.allocated_sets == sub_pool.max_sets {
                    sub_pool
                        .max_sets
                        .checked_mul(COUNT_BASE)
                        .expect("descriptor set pool count overflow")
                } else {
                    sub_pool.max_sets
                };
                let per_set = sub_pool.per_set.max(layout.descriptor_counts);
                let sub_pool = self.create_descriptor_sub_pool(max_sets, per_set);
                pool.sub_pools.insert(0, sub_pool);
            }

            let descriptor_set_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool.sub_pools[0].raw)
                .set_layouts(&descriptor_set_layouts);
            match unsafe { self.core.allocate_descriptor_sets(&descriptor_set_info) } {
                Ok(vk_sets) => {
                    pool.sub_pools[0].allocated_sets += 1;
                    return vk_sets[0];
                }
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY)
                | Err(vk::Result::ERROR_FRAGMENTED_POOL) => {}
                Err(other) => panic!("Unexpected descriptor allocation error: {:?}", other),
            };

            let sub_pool = &pool.sub_pools[0];
            let next_max_sets = sub_pool
                .max_sets
                .checked_mul(COUNT_BASE)
                .expect("descriptor set pool count overflow");
            let per_set = sub_pool.per_set.max(layout.descriptor_counts);
            let sub_pool = self.create_descriptor_sub_pool(next_max_sets, per_set);
            pool.sub_pools.insert(0, sub_pool);
        }
    }

    pub(super) fn reset_descriptor_pool(&self, pool: &mut DescriptorPool) {
        for sub_pool in pool.sub_pools.drain(1..) {
            unsafe {
                self.core.destroy_descriptor_pool(sub_pool.raw, None);
            }
        }

        let sub_pool = &mut pool.sub_pools[0];
        unsafe {
            self.core
                .reset_descriptor_pool(sub_pool.raw, vk::DescriptorPoolResetFlags::empty())
                .unwrap();
        }
        sub_pool.allocated_sets = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_counts_accumulate_binding_arrays() {
        let mut counts = DescriptorCounts::default();
        counts.add(vk::DescriptorType::STORAGE_BUFFER, 2);
        counts.add(vk::DescriptorType::STORAGE_BUFFER, 64);
        counts.add(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, 64);

        assert_eq!(counts.storage_buffers, 66);
        assert_eq!(counts.acceleration_structures, 64);
    }

    #[test]
    fn a_pool_budget_must_cover_every_descriptor_per_set() {
        let small = DescriptorCounts {
            storage_buffers: 1,
            acceleration_structures: 1,
            ..DescriptorCounts::default()
        };
        let scene = DescriptorCounts {
            storage_buffers: 322,
            acceleration_structures: 64,
            ..DescriptorCounts::default()
        };
        let budget = small.max(scene);

        assert!(budget.supports(scene));
        assert!(!small.supports(scene));
        assert_eq!(budget.storage_buffers.checked_mul(COUNT_BASE), Some(5152));
        assert_eq!(
            budget.acceleration_structures.checked_mul(COUNT_BASE),
            Some(1024)
        );
    }
}
