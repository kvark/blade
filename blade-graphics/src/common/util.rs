impl super::ComputePipeline {
    /// Return the dispatch group counts sufficient to cover the given extent.
    pub fn get_dispatch_for(&self, extent: crate::Extent) -> [u32; 3] {
        let wg_size = self.get_workgroup_size();
        [
            (extent.width + wg_size[0] - 1) / wg_size[0],
            (extent.height + wg_size[1] - 1) / wg_size[1],
            (extent.depth + wg_size[2] - 1) / wg_size[2],
        ]
    }
}
