/// Supported explicit full-subgroup compute pipelines. Absence of this
/// capability must not be replaced with the device's default subgroup width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FullSubgroups {
    pub min_size: u32,
    pub max_size: u32,
    pub max_workgroup_subgroups: u32,
}

impl FullSubgroups {
    /// The normal workgroup-size/device limits are checked independently.
    pub fn supports(self, size: u32, workgroup: [u32; 3]) -> bool {
        if !size.is_power_of_two()
            || size < self.min_size
            || size > self.max_size
            || workgroup.contains(&0)
            || !workgroup[0].is_multiple_of(size)
        {
            return false;
        }
        let invocations = workgroup
            .into_iter()
            .try_fold(1u64, |n, axis| n.checked_mul(u64::from(axis)));
        invocations.is_some_and(|n| n <= u64::from(size) * u64::from(self.max_workgroup_subgroups))
    }
}

#[cfg(test)]
mod tests {
    use super::FullSubgroups;

    #[test]
    fn explicit_full_subgroup_contract() {
        let caps = FullSubgroups {
            min_size: 16,
            max_size: 64,
            max_workgroup_subgroups: 8,
        };
        for size in [16, 32, 64] {
            assert!(caps.supports(size, [128, 1, 1]));
            assert!(caps.supports(size, [size, 2, 2]));
            assert!(caps.supports(size, [size * 8, 1, 1]));
            assert!(!caps.supports(size, [size * 9, 1, 1]));
            assert!(!caps.supports(size, [size / 2, 2, 1]));
            assert!(!caps.supports(size, [size, 0, 1]));
        }
        for size in [0, 1, 8, 24, 48, 128, u32::MAX] {
            assert!(!caps.supports(size, [128, 1, 1]));
        }
        assert!(!caps.supports(32, [u32::MAX - 31, u32::MAX, u32::MAX]));
        assert!(
            !FullSubgroups {
                max_workgroup_subgroups: 0,
                ..caps
            }
            .supports(32, [128, 1, 1])
        );
    }
}
