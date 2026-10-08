use ash::vk;
use std::collections;

//TODO: replace by an abstraction in `gpu-descriptor`
// https://github.com/zakarumych/gpu-descriptor/issues/42
const MAX_SETS_PER_POOL: usize = 64;

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
}

#[derive(Debug, Default)]
pub struct DescriptorPool {
    sets: collections::HashMap<super::UniqueLayoutId, DescriptorSetCache>,
}

#[derive(Debug, Default)]
struct DescriptorSetCache {
    sub_pools: Vec<vk::DescriptorPool>,
    sets: Vec<vk::DescriptorSet>,
    used: usize,
}

impl super::Device {
    fn create_descriptor_sub_pool(
        &self,
        max_sets: u32,
        per_set: DescriptorCounts,
    ) -> vk::DescriptorPool {
        log::info!("Creating a descriptor pool for at most {} sets", max_sets);
        // Each pool serves one layout, so reserve only the descriptors it uses.
        let pool_count = |count: u32| {
            count
                .checked_mul(max_sets)
                .expect("Descriptor pool size overflow")
        };
        let descriptor_sizes: Vec<_> = [
            (vk::DescriptorType::STORAGE_BUFFER, per_set.storage_buffers),
            (vk::DescriptorType::SAMPLED_IMAGE, per_set.sampled_images),
            (vk::DescriptorType::SAMPLER, per_set.samplers),
            (vk::DescriptorType::STORAGE_IMAGE, per_set.storage_images),
            (
                vk::DescriptorType::INLINE_UNIFORM_BLOCK_EXT,
                per_set.inline_uniform_bytes,
            ),
            (vk::DescriptorType::UNIFORM_BUFFER, per_set.uniform_buffers),
            (
                vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                per_set.acceleration_structures,
            ),
        ]
        .into_iter()
        .filter(|&(_, count)| count != 0)
        .map(|(ty, count)| vk::DescriptorPoolSize {
            ty,
            descriptor_count: pool_count(count),
        })
        .collect();

        let mut inline_uniform_block_info = vk::DescriptorPoolInlineUniformBlockCreateInfoEXT {
            max_inline_uniform_block_bindings: pool_count(per_set.inline_uniform_bindings),
            ..Default::default()
        };

        let mut descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .flags(self.workarounds.extra_descriptor_pool_create_flags)
            .pool_sizes(&descriptor_sizes);
        if per_set.inline_uniform_bindings != 0 {
            descriptor_pool_info = descriptor_pool_info.push_next(&mut inline_uniform_block_info);
        }

        unsafe {
            self.core
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap()
        }
    }

    pub(super) fn destroy_descriptor_pool(&self, pool: &mut DescriptorPool) {
        for (_, mut cache) in pool.sets.drain() {
            self.destroy_descriptor_cache(&mut cache);
        }
    }

    fn destroy_descriptor_cache(&self, cache: &mut DescriptorSetCache) {
        for raw in cache.sub_pools.drain(..) {
            unsafe { self.core.destroy_descriptor_pool(raw, None) };
        }
    }

    pub(super) fn allocate_descriptor_set(
        &self,
        pool: &mut DescriptorPool,
        layout: &super::DescriptorSetLayout,
    ) -> vk::DescriptorSet {
        let cache = pool.sets.entry(layout.unique_id).or_default();
        if cache.used == cache.sets.len() {
            self.grow_descriptor_cache(cache, layout);
        }
        let set = cache.sets[cache.used];
        cache.used += 1;
        set
    }

    fn grow_descriptor_cache(
        &self,
        cache: &mut DescriptorSetCache,
        layout: &super::DescriptorSetLayout,
    ) {
        // Start at one set and cap growth to limit spare capacity above the
        // highest number of sets used in a recording.
        let count = (cache.sets.len() + 1).min(MAX_SETS_PER_POOL);
        let raw = self.create_descriptor_sub_pool(count as u32, layout.descriptor_counts);
        let layouts = [layout.raw; MAX_SETS_PER_POOL];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(raw)
            .set_layouts(&layouts[..count]);
        // Allocate the whole pool at once: earlier pools are full and never
        // need to be searched again. Destroy this pool if allocation fails.
        let sets = unsafe { self.core.allocate_descriptor_sets(&info) }.unwrap_or_else(|error| {
            unsafe { self.core.destroy_descriptor_pool(raw, None) };
            panic!("Unable to allocate descriptor sets: {error:?}");
        });
        cache.sub_pools.push(raw);
        cache.sets.extend(sets);
    }

    pub(super) fn reset_descriptor_pool(&self, pool: &mut DescriptorPool) {
        // The command buffer is no longer in flight. Each layout has its own
        // pools, so reclaiming an unused cache leaves the other sets intact.
        pool.sets.retain(|_, cache| {
            if cache.used == 0 {
                self.destroy_descriptor_cache(cache);
                false
            } else {
                cache.used = 0;
                true
            }
        });
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
}
