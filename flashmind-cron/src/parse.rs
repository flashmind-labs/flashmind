//! Nom-based POSIX 5-field cron expression parser.
//!
//! Supports wildcards (`*`), steps (`*/N`, `range/N`), ranges (`N-M`),
//! comma-separated lists, named months (JAN-DEC), and named days (MON-SUN).

use nom::IResult;
use nom::Parser;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::{char, digit1, multispace0, multispace1};
use nom::combinator::{map, map_res};
use nom::multi::separated_list1;
use nom::sequence::preceded;

use crate::error::CronError;

// ---------------------------------------------------------------------------
// FieldSet — bitmap for one cron field
// ---------------------------------------------------------------------------

/// Bitmap representing which values are active for a single cron field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldSet {
    bits: Vec<bool>,
    offset: u32,
}

impl FieldSet {
    fn new(min: u32, max: u32) -> Self {
        Self {
            bits: vec![false; (max - min + 1) as usize],
            offset: min,
        }
    }

    fn all(min: u32, max: u32) -> Self {
        Self {
            bits: vec![true; (max - min + 1) as usize],
            offset: min,
        }
    }

    fn set(&mut self, val: u32) {
        let idx = (val - self.offset) as usize;
        if idx < self.bits.len() {
            self.bits[idx] = true;
        }
    }

    /// Returns true if the given value is active in this field.
    pub fn contains(&self, val: u32) -> bool {
        let idx = val.wrapping_sub(self.offset) as usize;
        idx < self.bits.len() && self.bits[idx]
    }

    /// Returns true if every value in the valid range is set.
    pub fn is_all(&self) -> bool {
        self.bits.iter().all(|&b| b)
    }

    /// Minimum value in the valid range (inclusive).
    pub fn min_val(&self) -> u32 {
        self.offset
    }

    /// Maximum value in the valid range (inclusive).
    pub fn max_val(&self) -> u32 {
        self.offset + self.bits.len() as u32 - 1
    }

    /// Iterator over all active values.
    pub fn iter_set(&self) -> impl Iterator<Item = u32> + '_ {
        self.bits
            .iter()
            .enumerate()
            .filter(|(_, b)| **b)
            .map(move |(i, _)| i as u32 + self.offset)
    }
}

// ---------------------------------------------------------------------------
// CronExpr — parsed 5-field cron expression
// ---------------------------------------------------------------------------

/// Parsed representation of a POSIX 5-field cron expression.
#[derive(Debug, Clone)]
pub struct CronExpr {
    pub minute: FieldSet,
    pub hour: FieldSet,
    pub day_of_month: FieldSet,
    pub month: FieldSet,
    pub day_of_week: FieldSet,
}

// ---------------------------------------------------------------------------
// Named value lookups
// ---------------------------------------------------------------------------

fn month_name(s: &str) -> Option<u32> {
    match s.to_ascii_uppercase().as_str() {
        "JAN" => Some(1),
        "FEB" => Some(2),
        "MAR" => Some(3),
        "APR" => Some(4),
        "MAY" => Some(5),
        "JUN" => Some(6),
        "JUL" => Some(7),
        "AUG" => Some(8),
        "SEP" => Some(9),
        "OCT" => Some(10),
        "NOV" => Some(11),
        "DEC" => Some(12),
        _ => None,
    }
}

fn day_name(s: &str) -> Option<u32> {
    match s.to_ascii_uppercase().as_str() {
        "SUN" => Some(0),
        "MON" => Some(1),
        "TUE" => Some(2),
        "WED" => Some(3),
        "THU" => Some(4),
        "FRI" => Some(5),
        "SAT" => Some(6),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Nom parsers
// ---------------------------------------------------------------------------

fn parse_u32(input: &str) -> IResult<&str, u32> {
    map_res(digit1, |s: &str| s.parse::<u32>()).parse(input)
}

fn parse_name_or_num<'a>(
    name_fn: fn(&str) -> Option<u32>,
) -> impl FnMut(&'a str) -> IResult<&'a str, u32> {
    move |input: &'a str| {
        alt((
            parse_u32,
            map_res(
                nom::bytes::complete::take_while1(|c: char| c.is_ascii_alphabetic()),
                move |s: &str| name_fn(s).ok_or("unknown name"),
            ),
        ))
        .parse(input)
    }
}

/// Single value, range, or range-with-step.
fn parse_atom<'a>(
    min: u32,
    max: u32,
    name_fn: fn(&str) -> Option<u32>,
) -> impl FnMut(&'a str) -> IResult<&'a str, FieldSet> {
    move |input: &'a str| {
        alt((
            // */step
            map(preceded(tag("*/"), parse_u32), |step| {
                let mut fs = FieldSet::new(min, max);
                let step = step.max(1);
                let mut v = min;
                while v <= max {
                    fs.set(v);
                    v += step;
                }
                fs
            }),
            // wildcard
            map(char('*'), |_| FieldSet::all(min, max)),
            // range/step or range or single
            {
                let mut val_parser = parse_name_or_num(name_fn);
                move |input: &'a str| {
                    let (input, start) = val_parser(input)?;
                    if let Ok((input, end)) = preceded(char('-'), &mut val_parser).parse(input) {
                        // range with optional step
                        if let Ok((input, step)) = preceded(char('/'), parse_u32).parse(input) {
                            let mut fs = FieldSet::new(min, max);
                            let step = step.max(1);
                            let mut v = start;
                            while v <= end {
                                fs.set(v);
                                v += step;
                            }
                            Ok((input, fs))
                        } else {
                            let mut fs = FieldSet::new(min, max);
                            for v in start..=end {
                                fs.set(v);
                            }
                            Ok((input, fs))
                        }
                    } else {
                        let mut fs = FieldSet::new(min, max);
                        fs.set(start);
                        Ok((input, fs))
                    }
                }
            },
        ))
        .parse(input)
    }
}

/// Comma-separated list of atoms, merged into one FieldSet.
fn parse_field<'a>(
    min: u32,
    max: u32,
    name_fn: fn(&str) -> Option<u32>,
) -> impl FnMut(&'a str) -> IResult<&'a str, FieldSet> {
    move |input: &'a str| {
        let (input, sets) =
            separated_list1(char(','), parse_atom(min, max, name_fn)).parse(input)?;
        let mut merged = FieldSet::new(min, max);
        for s in &sets {
            for i in 0..merged.bits.len() {
                merged.bits[i] = merged.bits[i] || s.bits[i];
            }
        }
        Ok((input, merged))
    }
}

fn no_names(_: &str) -> Option<u32> {
    None
}

fn parse_cron_expr(input: &str) -> IResult<&str, CronExpr> {
    let (input, _) = multispace0(input)?;
    let (input, minute) = parse_field(0, 59, no_names).parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, hour) = parse_field(0, 23, no_names).parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, day_of_month) = parse_field(1, 31, no_names).parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, month) = parse_field(1, 12, month_name).parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, day_of_week) = parse_field(0, 7, day_name).parse(input)?;
    // Normalize: day_of_week 7 (Sunday) maps to 0
    let mut dow = day_of_week;
    if dow.contains(7) {
        dow.set(0);
    }
    let (input, _) = multispace0(input)?;
    Ok((
        input,
        CronExpr {
            minute,
            hour,
            day_of_month,
            month,
            day_of_week: dow,
        },
    ))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a POSIX 5-field cron expression string.
pub fn parse_cron(input: &str) -> Result<CronExpr, CronError> {
    match parse_cron_expr(input) {
        Ok(("", expr)) => Ok(expr),
        Ok((rest, _)) => Err(CronError::InvalidExpression(format!(
            "unexpected trailing input: '{rest}'"
        ))),
        Err(e) => Err(CronError::InvalidExpression(format!("{e}"))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_every_minute() {
        let expr = parse_cron("* * * * *").unwrap();
        assert!(expr.minute.is_all());
        assert!(expr.hour.is_all());
        assert!(expr.day_of_month.is_all());
        assert!(expr.month.is_all());
        assert!(expr.day_of_week.is_all());
    }

    #[test]
    fn test_specific_values() {
        let expr = parse_cron("5 14 1 6 3").unwrap();
        assert!(expr.minute.contains(5));
        assert!(!expr.minute.contains(4));
        assert!(expr.hour.contains(14));
        assert!(expr.day_of_month.contains(1));
        assert!(expr.month.contains(6));
        assert!(expr.day_of_week.contains(3));
    }

    #[test]
    fn test_ranges() {
        let expr = parse_cron("1-5 * * * *").unwrap();
        for v in 1..=5 {
            assert!(expr.minute.contains(v));
        }
        assert!(!expr.minute.contains(0));
        assert!(!expr.minute.contains(6));
    }

    #[test]
    fn test_steps() {
        let expr = parse_cron("*/15 * * * *").unwrap();
        assert!(expr.minute.contains(0));
        assert!(expr.minute.contains(15));
        assert!(expr.minute.contains(30));
        assert!(expr.minute.contains(45));
        assert!(!expr.minute.contains(1));
    }

    #[test]
    fn test_range_with_step() {
        let expr = parse_cron("1-10/3 * * * *").unwrap();
        assert!(expr.minute.contains(1));
        assert!(expr.minute.contains(4));
        assert!(expr.minute.contains(7));
        assert!(expr.minute.contains(10));
        assert!(!expr.minute.contains(2));
    }

    #[test]
    fn test_list() {
        let expr = parse_cron("1,15,30 * * * *").unwrap();
        assert!(expr.minute.contains(1));
        assert!(expr.minute.contains(15));
        assert!(expr.minute.contains(30));
        assert!(!expr.minute.contains(2));
    }

    #[test]
    fn test_named_months() {
        let expr = parse_cron("0 0 1 JAN,MAR,DEC *").unwrap();
        assert!(expr.month.contains(1));
        assert!(expr.month.contains(3));
        assert!(expr.month.contains(12));
        assert!(!expr.month.contains(2));
    }

    #[test]
    fn test_named_days() {
        let expr = parse_cron("0 0 * * MON-FRI").unwrap();
        for d in 1..=5 {
            assert!(expr.day_of_week.contains(d));
        }
        assert!(!expr.day_of_week.contains(0));
        assert!(!expr.day_of_week.contains(6));
    }

    #[test]
    fn test_named_days_case_insensitive() {
        let expr = parse_cron("0 0 * * mon,Wed,FRI").unwrap();
        assert!(expr.day_of_week.contains(1));
        assert!(expr.day_of_week.contains(3));
        assert!(expr.day_of_week.contains(5));
    }

    #[test]
    fn test_sunday_7_maps_to_0() {
        let expr = parse_cron("0 0 * * 7").unwrap();
        assert!(expr.day_of_week.contains(0));
    }

    #[test]
    fn test_invalid_expression() {
        assert!(parse_cron("not a cron").is_err());
        assert!(parse_cron("* * *").is_err());
        assert!(parse_cron("").is_err());
    }

    #[test]
    fn test_complex_expression() {
        let expr = parse_cron("0,30 9-17 1,15 JAN-JUN MON-FRI").unwrap();
        assert!(expr.minute.contains(0));
        assert!(expr.minute.contains(30));
        assert!(!expr.minute.contains(15));
        for h in 9..=17 {
            assert!(expr.hour.contains(h));
        }
        assert!(!expr.hour.contains(8));
        assert!(expr.day_of_month.contains(1));
        assert!(expr.day_of_month.contains(15));
        for m in 1..=6 {
            assert!(expr.month.contains(m));
        }
        assert!(!expr.month.contains(7));
    }
}
