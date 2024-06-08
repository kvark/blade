use ash::vk;

//TODO: replace by an abstraction in `gpu-descriptor`
// https://github.com/zakarumych/gpu-descriptor/issues/42
const COUNT_BASE: u32 = 16;

#[derive(Debug)]
pub struct DescriptorPool {
    sub_pools: Vec<vk::DescriptorPool>,
    current_pool: usize,
}

impl super::Device {
    fn create_descriptor_sub_pool(&self, max_sets: u32) -> vk::DescriptorPool {
        log::info!("Creating a descriptor pool for at most {} sets", max_sets);
        let mut descriptor_sizes = vec![
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::INLINE_UNIFORM_BLOCK_EXT,
                descriptor_count: max_sets * crate::limits::PLAIN_DATA_SIZE,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: max_sets,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 2 * max_sets,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: max_sets,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: max_sets,
            },
        ];
        if self.ray_tracing.is_some() {
            descriptor_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count: max_sets,
            });
        }

        let mut inline_uniform_block_info = vk::DescriptorPoolInlineUniformBlockCreateInfoEXT {
            max_inline_uniform_block_bindings: max_sets,
            ..Default::default()
        };
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(&descriptor_sizes)
            .push_next(&mut inline_uniform_block_info);
        unsafe {
            self.core
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap()
        }
    }

    pub(super) fn create_descriptor_pool(&self) -> DescriptorPool {
        let vk_pool = self.create_descriptor_sub_pool(COUNT_BASE);
        DescriptorPool {
            sub_pools: vec![vk_pool],
            current_pool: 0,
        }
    }

    pub(super) fn destroy_descriptor_pool(&self, pool: &mut DescriptorPool) {
        for sub_pool in pool.sub_pools.drain(..) {
            unsafe { self.core.destroy_descriptor_pool(sub_pool, None) };
        }
    }

    pub(super) fn allocate_descriptor_set(
        &self,
        pool: &mut DescriptorPool,
        layout: &super::DescriptorSetLayout,
    ) -> vk::DescriptorSet {
        let descriptor_set_layouts = [layout.raw];

        while pool.current_pool < pool.sub_pools.len() {
            let descriptor_set_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool.sub_pools[pool.current_pool])
                .set_layouts(&descriptor_set_layouts);
            let result = unsafe { self.core.allocate_descriptor_sets(&descriptor_set_info) };
            match result {
                Ok(vk_sets) => return vk_sets[0],
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY)
                | Err(vk::Result::ERROR_FRAGMENTED_POOL) => {
                    pool.current_pool += 1;
                }
                Err(other) => panic!("Unexpected descriptor allocation error: {:?}", other),
            };
        }

        let next_max_sets = COUNT_BASE.pow(pool.sub_pools.len() as u32 + 1);
        let vk_pool = self.create_descriptor_sub_pool(next_max_sets);
        pool.sub_pools.push(vk_pool);
        pool.current_pool = pool.sub_pools.len() - 1;

        let descriptor_set_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool.sub_pools[pool.current_pool])
            .set_layouts(&descriptor_set_layouts);

        let vk_sets = unsafe {
            self.core
                .allocate_descriptor_sets(&descriptor_set_info)
                .unwrap()
        };
        vk_sets[0]
    }

    pub(super) fn reset_descriptor_pool(&self, pool: &mut DescriptorPool) {
        for &vk_pool in pool.sub_pools.iter() {
            unsafe {
                self.core
                    .reset_descriptor_pool(vk_pool, vk::DescriptorPoolResetFlags::empty())
                    .unwrap();
            }
        }
        pool.current_pool = 0
    }
}
