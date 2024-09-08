use ash::vk;

const fn pool_size(i: usize) -> u32 {
    const COUNT_BASE: u32 = 4;
    COUNT_BASE.pow(i as u32 + 1)
}

#[derive(Debug)]
pub(super) struct QueryPool {
    sub_pools: Vec<vk::QueryPool>,
    count: u32,
}

impl super::Device {
    fn create_query_sub_pool(&self, max_queries: u32) -> vk::QueryPool {
        log::info!("Creating a query set for at most {} queries", max_queries);

        let query_pool_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(max_queries);

        unsafe {
            let vk_pool = self.core.create_query_pool(&query_pool_info, None).unwrap();
            self.core.reset_query_pool(vk_pool, 0, max_queries);
            vk_pool
        }
    }

    pub(super) fn create_query_pool(&self) -> QueryPool {
        QueryPool {
            sub_pools: match self.timing {
                Some(_) => {
                    let pool_size = pool_size(0);
                    let vk_pool = self.create_query_sub_pool(pool_size);
                    vec![vk_pool]
                }
                None => Vec::new(),
            },
            count: 0,
        }
    }

    pub(super) fn destroy_query_pool(&self, pool: &mut QueryPool) {
        for sub_pool in pool.sub_pools.drain(..) {
            unsafe { self.core.destroy_query_pool(sub_pool, None) };
        }
    }

    pub(super) fn allocate_query(&self, pool: &mut QueryPool) -> (vk::QueryPool, u32) {
        let mut remaining = pool.count;
        pool.count += 1;
        for (i, &vk_pool) in pool.sub_pools.iter().enumerate() {
            let pool_size = pool_size(i);
            if remaining < pool_size {
                return (vk_pool, remaining);
            }
            remaining -= pool_size;
        }
        let next_max_queries = pool_size(pool.sub_pools.len());
        let vk_pool = self.create_query_sub_pool(next_max_queries);
        pool.sub_pools.push(vk_pool);
        (vk_pool, 0)
    }

    pub(super) fn reset_query_pool(&self, pool: &mut QueryPool) {
        for (i, &vk_pool) in pool.sub_pools.iter().enumerate() {
            let pool_size = pool_size(i);
            unsafe {
                self.core.reset_query_pool(vk_pool, 0, pool_size);
            }
        }
        pool.count = 0;
    }

    pub(super) fn get_query_pool_results(&self, pool: &QueryPool) -> Vec<u64> {
        let mut timestamps = Vec::new();
        let mut remaining = pool.count;
        for (i, &vk_pool) in pool.sub_pools.iter().enumerate() {
            if remaining == 0 {
                break;
            }
            let pool_size = pool_size(i);
            let count = remaining.min(pool_size);
            remaining -= count;
            let base = timestamps.len();
            timestamps.resize(base + count as usize, 0);
            unsafe {
                self.core
                    .get_query_pool_results(
                        vk_pool,
                        0,
                        &mut timestamps[base..],
                        vk::QueryResultFlags::TYPE_64,
                    )
                    .unwrap();
            }
        }
        timestamps
    }
}
