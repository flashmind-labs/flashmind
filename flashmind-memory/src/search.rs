//! Reciprocal Rank Fusion (RRF) for hybrid search.
//!
//! # Why hybrid search?
//!
//! Vector search finds semantically similar results (synonyms, paraphrases) but
//! misses exact keyword matches (URLs, error codes, proper nouns). FTS5 keyword
//! search finds exact matches but misses semantic similarity. Hybrid search runs
//! both, then fuses the results.
//!
//! # How RRF works
//!
//! Given two ranked lists (one from vector search, one from FTS5):
//!
//! ```text
//! Vector search: [A, B, C, D]    (ranked by cosine similarity)
//! FTS5 search:   [C, E, A, F]    (ranked by BM25 score)
//! ```
//!
//! For each document, compute: `score = Σ 1/(k + rank)` across all lists where
//! it appears. With k=10:
//!
//! ```text
//! A: 1/(10+0) + 1/(10+2) = 0.10 + 0.083 = 0.183  ← appears in both, boosted
//! C: 1/(10+2) + 1/(10+0) = 0.083 + 0.10 = 0.183  ← appears in both, boosted
//! B: 1/(10+1)                                = 0.091  ← only in vector
//! E: 1/(10+1)                                = 0.091  ← only in FTS
//! ```
//!
//! Documents that rank high in **both** lists get the highest fused scores.
//! The `k` parameter (default 10) controls how much top ranks are favored —
//! lower k = more discrimination between top and bottom ranks.
//!
//! # Re-ranking by cosine similarity
//!
//! RRF fusion selects good candidates from both search methods, but the RRF scores
//! don't directly reflect relevance. After fusion, `search_hybrid` re-ranks candidates
//! by computing actual cosine similarity between the query and stored embeddings.
//! This returns real similarity scores (0-1) instead of normalized RRF scores.
//!
//! # Usage in flashmind_memory
//!
//! `VectorMemory::search_hybrid()` runs:
//! 1. sqlite-vec vector search → ranked ID list (3x over-retrieve)
//! 2. FTS5 MATCH query → ranked ID list (3x over-retrieve)
//! 3. `rrf_fuse([vec_ids, fts_ids], 10.0)` → fused scores (candidate selection)
//! 4. Over-retrieve top 3x candidates
//! 5. Fetch stored embeddings and compute cosine similarity
//! 6. Sort by cosine similarity descending → final results

use std::{cmp::Ordering, collections::HashMap};

/// Reciprocal Rank Fusion over multiple ranked ID lists.
///
/// For each ID across all lists: `score = sum(1 / (k + rank))` where rank is 0-indexed.
/// Returns (id, score) pairs sorted by score descending.
pub fn rrf_fuse(lists: &[Vec<&str>], k: f32) -> Vec<(String, f32)> {
    let mut scores: HashMap<&str, f32> = HashMap::new();

    for list in lists {
        for (rank, &id) in list.iter().enumerate() {
            *scores.entry(id).or_insert(0.0) += 1.0 / (k + rank as f32);
        }
    }

    let mut results: Vec<(String, f32)> = scores
        .into_iter()
        .map(|(id, score)| (id.to_string(), score))
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    results
}

/// Normalize scores in-place to the [0, 1] range by dividing by the maximum score.
///
/// Handles empty input and zero maximum gracefully (no-op in both cases).
pub fn normalize_scores(results: &mut [(String, f32)]) {
    let max = results
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);

    if results.is_empty() || max == 0.0 || !max.is_finite() {
        return;
    }

    for (_, score) in results.iter_mut() {
        *score /= max;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_single_list() {
        let list = vec!["a", "b", "c"];
        let results = rrf_fuse(&[list], 60.0);

        let score_of = |id: &str| results.iter().find(|(s, _)| s == id).map(|(_, v)| *v);

        assert!(score_of("a").unwrap() > score_of("b").unwrap());
        assert!(score_of("b").unwrap() > score_of("c").unwrap());
    }

    #[test]
    fn test_rrf_two_lists_boost() {
        let list1 = vec!["x", "y", "z"];
        let list2 = vec!["x", "a", "b"];
        let results = rrf_fuse(&[list1, list2], 60.0);

        let score_of = |id: &str| results.iter().find(|(s, _)| s == id).map(|(_, v)| *v);

        // "x" is first in both lists, so it should have the highest score
        let x_score = score_of("x").unwrap();
        for (id, score) in &results {
            if id != "x" {
                assert!(x_score > *score, "x should beat {id}");
            }
        }
    }

    #[test]
    fn test_normalize_scores() {
        let mut results = vec![
            ("a".to_string(), 0.5_f32),
            ("b".to_string(), 1.0_f32),
            ("c".to_string(), 0.25_f32),
        ];

        normalize_scores(&mut results);

        let max = results
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!((max - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalize_empty() {
        let mut results: Vec<(String, f32)> = vec![];
        normalize_scores(&mut results); // must not panic
    }
}
