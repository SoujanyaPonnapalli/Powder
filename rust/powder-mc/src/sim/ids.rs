//! String interning for node IDs and region names.
//!
//! Python keys events, cancellation tables and cluster membership by string.
//! Hashing those strings on every event is pure overhead, so the port interns
//! every identifier to a `Sym` (a `u32`) once and compares integers
//! thereafter.  Event cancellation tables become flat vectors indexed by
//! symbol instead of string-keyed hash maps.
//!
//! The interner lives inside [`ClusterState`](crate::sim::cluster::ClusterState)
//! behind a `RefCell` so that protocols and strategies, which only ever hold
//! `&ClusterState`, can still mint identifiers for nodes they are about to
//! create.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// An interned identifier.  Cheap to copy, compare and index with.
pub type Sym = u32;

/// Pre-interned symbol for the Raft protocol's own event target.
pub const SYM_PROTOCOL: Sym = 0;
/// Pre-interned symbol for cluster-wide (reconfiguration) events.
pub const SYM_CLUSTER: Sym = 1;

const RESERVED: [&str; 2] = ["protocol", "cluster"];

/// The forward and reverse tables.
///
/// Both sides hold the same `Rc<str>`, so a name costs one allocation
/// rather than two.
#[derive(Debug, Default)]
struct Inner {
    names: Vec<Rc<str>>,
    lookup: HashMap<Rc<str>, Sym>,
}

/// Append-only bidirectional map between identifier strings and [`Sym`]s.
#[derive(Debug)]
pub struct Interner {
    inner: RefCell<Inner>,
}

impl Default for Interner {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Interner {
    fn clone(&self) -> Self {
        Interner {
            inner: RefCell::new(Inner {
                names: self.inner.borrow().names.clone(),
                lookup: self.inner.borrow().lookup.clone(),
            }),
        }
    }
}

impl Interner {
    /// Create an interner with the reserved pseudo-targets already assigned
    /// to [`SYM_PROTOCOL`] and [`SYM_CLUSTER`].
    pub fn new() -> Self {
        let interner = Interner {
            inner: RefCell::new(Inner::default()),
        };
        for name in RESERVED {
            interner.intern(name);
        }
        interner
    }

    /// Return the symbol for `name`, assigning a fresh one if needed.
    pub fn intern(&self, name: &str) -> Sym {
        let mut inner = self.inner.borrow_mut();
        if let Some(&sym) = inner.lookup.get(name) {
            return sym;
        }
        let sym = inner.names.len() as Sym;
        let shared: Rc<str> = Rc::from(name);
        inner.names.push(Rc::clone(&shared));
        inner.lookup.insert(shared, sym);
        sym
    }

    /// Return the symbol for `name` if it has already been interned.
    pub fn get(&self, name: &str) -> Option<Sym> {
        self.inner.borrow().lookup.get(name).copied()
    }

    /// Resolve a symbol back to its string form.
    ///
    /// Only used for reporting and JSON output, so the allocation is fine.
    pub fn name(&self, sym: Sym) -> String {
        self.inner
            .borrow()
            .names
            .get(sym as usize)
            .map(|n| n.to_string())
            .unwrap_or_else(|| format!("<unknown sym {sym}>"))
    }

    /// Compare two symbols by their string form, without allocating.
    ///
    /// Raft's leader election sorts candidates by node ID string, so the port
    /// needs name ordering -- which is not the same as symbol ordering.
    pub fn cmp_names(&self, a: Sym, b: Sym) -> std::cmp::Ordering {
        let inner = self.inner.borrow();
        let left: &str = inner.names.get(a as usize).map(|n| &**n).unwrap_or("");
        let right: &str = inner.names.get(b as usize).map(|n| &**n).unwrap_or("");
        left.cmp(right)
    }

    /// Number of distinct symbols assigned so far.
    pub fn len(&self) -> usize {
        self.inner.borrow().names.len()
    }

    /// Whether no symbols have been assigned.  Always false in practice,
    /// since the reserved targets are interned at construction.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_symbols_have_fixed_values() {
        let interner = Interner::new();
        assert_eq!(interner.get("protocol"), Some(SYM_PROTOCOL));
        assert_eq!(interner.get("cluster"), Some(SYM_CLUSTER));
        assert_eq!(interner.len(), 2);
    }

    #[test]
    fn interning_is_stable_and_reversible() {
        let interner = Interner::new();
        let a = interner.intern("node_0");
        let b = interner.intern("node_1");
        assert_ne!(a, b);
        assert_eq!(interner.intern("node_0"), a);
        assert_eq!(interner.name(a), "node_0");
        assert_eq!(interner.name(b), "node_1");
    }

    #[test]
    fn get_returns_none_for_unknown_names() {
        let interner = Interner::new();
        assert_eq!(interner.get("nope"), None);
    }

    #[test]
    fn cmp_names_orders_lexicographically() {
        use std::cmp::Ordering;
        let interner = Interner::new();
        let ten = interner.intern("node_10");
        let two = interner.intern("node_2");
        // Symbol order says node_10 first; name order says node_2 first.
        assert!(ten < two);
        assert_eq!(interner.cmp_names(ten, two), Ordering::Less);
        assert_eq!(interner.cmp_names(two, ten), Ordering::Greater);
        assert_eq!(interner.cmp_names(two, two), Ordering::Equal);
    }

    #[test]
    fn clone_preserves_assignments() {
        let interner = Interner::new();
        let sym = interner.intern("node_0");
        let copy = interner.clone();
        assert_eq!(copy.get("node_0"), Some(sym));
        // The clone is independent: new names do not leak back.
        copy.intern("node_1");
        assert_eq!(interner.get("node_1"), None);
    }
}
