/// A small ring of optional resources, one slot per in-flight generation.
pub(crate) struct RollingBuffer<T, const N: usize> {
    slots: [Option<T>; N],
    index: usize,
}

impl<T, const N: usize> Default for RollingBuffer<T, N> {
    fn default() -> Self {
        assert!(N > 0);
        Self {
            slots: std::array::from_fn(|_| None),
            index: 0,
        }
    }
}

impl<T, const N: usize> RollingBuffer<T, N> {
    /// Advance the ring and return the slot to fill for this frame.
    pub fn next(&mut self) -> &mut Option<T> {
        self.next_indexed().1
    }

    /// Advance the ring and return both the generation index and its slot.
    pub fn next_indexed(&mut self) -> (usize, &mut Option<T>) {
        let i = self.index;
        self.index = (i + 1) % N;
        (i, &mut self.slots[i])
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.slots[index].as_ref()
    }

    /// The slot filled by the previous [`Self::next`] call.
    pub fn last(&self) -> Option<&T> {
        let i = (self.index + N - 1) % N;
        self.slots[i].as_ref()
    }

    /// Take every occupied slot and reset the ring.
    pub fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.index = 0;
        self.slots.iter_mut().filter_map(Option::take)
    }
}
