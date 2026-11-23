//! Undo / redo: a stack of scene snapshots (whatever text format the app
//! saves scenes in). Apps record a snapshot just before each change.

pub struct History {
    past: Vec<String>,
    future: Vec<String>,
    /// Most snapshots kept, and most bytes kept (oldest dropped first).
    max_count: usize,
    max_bytes: usize,
}

impl Default for History {
    fn default() -> Self {
        Self { past: Vec::new(), future: Vec::new(), max_count: 40, max_bytes: 64 << 20 }
    }
}

impl History {
    /// Remember `state` (the scene as it is before a change). Redo history
    /// is dropped: a new change starts a new branch.
    pub fn record(&mut self, state: String) {
        if self.past.last() == Some(&state) {
            return;
        }
        self.past.push(state);
        self.future.clear();
        self.trim();
    }

    /// Step back: returns the state to restore; `current` goes to redo.
    pub fn undo(&mut self, current: String) -> Option<String> {
        let prev = self.past.pop()?;
        self.future.push(current);
        Some(prev)
    }

    /// Step forward again: returns the state to restore; `current` goes to undo.
    pub fn redo(&mut self, current: String) -> Option<String> {
        let next = self.future.pop()?;
        self.past.push(current);
        self.trim();
        Some(next)
    }

    pub fn can_undo(&self) -> bool {
        !self.past.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.future.is_empty()
    }

    fn trim(&mut self) {
        let mut bytes: usize = self.past.iter().chain(&self.future).map(|s| s.len()).sum();
        while self.past.len() > 1 && (self.past.len() > self.max_count || bytes > self.max_bytes) {
            bytes -= self.past.remove(0).len();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::History;

    #[test]
    fn undo_redo_and_branching() {
        let mut h = History::default();
        h.record("a".into());
        h.record("b".into());
        assert_eq!(h.undo("c".into()).as_deref(), Some("b"));
        assert_eq!(h.undo("b".into()).as_deref(), Some("a"));
        assert_eq!(h.undo("a".into()), None);
        assert_eq!(h.redo("a".into()).as_deref(), Some("b"));
        h.record("c".into()); // a new change...
        assert!(!h.can_redo()); // ...drops the redo branch
        let mut h = History { max_count: 3, ..Default::default() };
        for k in 0..10 {
            h.record(k.to_string());
        }
        assert_eq!(h.undo("x".into()).as_deref(), Some("9"));
        assert_eq!(h.undo("9".into()).as_deref(), Some("8"));
        assert_eq!(h.undo("8".into()).as_deref(), Some("7"));
        assert_eq!(h.undo("7".into()), None);
    }
}
