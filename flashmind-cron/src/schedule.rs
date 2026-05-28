//! Cron schedule matching and next-fire-time computation.

use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, TimeZone, Timelike, Utc};
use chrono_tz::Tz;

use crate::error::CronError;
use crate::parse::{CronExpr, parse_cron};

// ---------------------------------------------------------------------------
// CronSchedule
// ---------------------------------------------------------------------------

/// A parsed cron schedule that can test whether a given time matches and
/// compute the next matching time after a given instant.
pub struct CronSchedule {
    expr: CronExpr,
}

impl CronSchedule {
    /// Parse a 5-field POSIX cron expression into a schedule.
    pub fn parse(s: &str) -> Result<Self, CronError> {
        Ok(Self {
            expr: parse_cron(s)?,
        })
    }

    /// Returns true if the given UTC datetime matches this cron schedule.
    ///
    /// POSIX semantics: when both day_of_month and day_of_week are restricted
    /// (not wildcards), the schedule fires if EITHER matches (OR logic).
    pub fn matches(&self, dt: &DateTime<Utc>) -> bool {
        self.matches_components(
            dt.minute(),
            dt.hour(),
            dt.day(),
            dt.month(),
            dt.weekday().num_days_from_sunday(),
        )
    }

    /// Returns true if the given UTC datetime, converted to the given timezone,
    /// matches this cron schedule.
    pub fn matches_in_tz(&self, utc: &DateTime<Utc>, tz: Tz) -> bool {
        let local = utc.with_timezone(&tz);
        self.matches_components(
            local.minute(),
            local.hour(),
            local.day(),
            local.month(),
            local.weekday().num_days_from_sunday(),
        )
    }

    /// Compute the next UTC datetime after `after` that matches this schedule.
    ///
    /// Returns `None` if no match is found within 2 years (guards against
    /// impossible schedules like Feb 31).
    pub fn next_after(&self, after: &DateTime<Utc>) -> Option<DateTime<Utc>> {
        let limit = *after + chrono::Duration::days(366 * 2);
        let mut dt = *after + chrono::Duration::minutes(1);
        // Snap to start of minute
        dt = Utc
            .with_ymd_and_hms(dt.year(), dt.month(), dt.day(), dt.hour(), dt.minute(), 0)
            .single()?;

        while dt <= limit {
            if !self.expr.month.contains(dt.month()) {
                dt = advance_month(dt)?;
                continue;
            }
            if !self.day_matches_components(dt.day(), dt.weekday().num_days_from_sunday()) {
                dt = advance_day(dt)?;
                continue;
            }
            if !self.expr.hour.contains(dt.hour()) {
                dt = advance_hour(dt)?;
                continue;
            }
            if self.expr.minute.contains(dt.minute()) {
                return Some(dt);
            }
            if let Some(next_min) = next_set_value(&self.expr.minute, dt.minute() + 1) {
                dt = Utc
                    .with_ymd_and_hms(dt.year(), dt.month(), dt.day(), dt.hour(), next_min, 0)
                    .single()?;
                if dt <= limit {
                    return Some(dt);
                }
            }
            dt = advance_hour(dt)?;
        }

        None
    }

    /// Compute the next UTC datetime after `after` that matches this schedule
    /// when evaluated in the given timezone. Handles DST correctly.
    pub fn next_after_in_tz(&self, after: &DateTime<Utc>, tz: Tz) -> Option<DateTime<Utc>> {
        let limit = *after + chrono::Duration::days(366 * 2);
        let local = after.with_timezone(&tz);
        let mut ndt = local.naive_local() + chrono::Duration::minutes(1);
        // Snap to start of minute
        ndt = NaiveDate::from_ymd_opt(ndt.year(), ndt.month(), ndt.day())?.and_hms_opt(
            ndt.hour(),
            ndt.minute(),
            0,
        )?;

        while to_utc(ndt, tz).map_or(true, |u| u <= limit) {
            if !self.expr.month.contains(ndt.month()) {
                ndt = advance_month_naive(ndt)?;
                continue;
            }
            if !self.day_matches_components(ndt.day(), ndt.weekday().num_days_from_sunday()) {
                ndt = advance_day_naive(ndt)?;
                continue;
            }
            if !self.expr.hour.contains(ndt.hour()) {
                ndt = advance_hour_naive(ndt)?;
                continue;
            }
            if self.expr.minute.contains(ndt.minute()) {
                if let Some(utc) = to_utc(ndt, tz) {
                    if utc > *after {
                        return Some(utc);
                    }
                }
                ndt = ndt + chrono::Duration::minutes(1);
                continue;
            }
            if let Some(next_min) = next_set_value(&self.expr.minute, ndt.minute() + 1) {
                let candidate = NaiveDate::from_ymd_opt(ndt.year(), ndt.month(), ndt.day())?
                    .and_hms_opt(ndt.hour(), next_min, 0)?;
                if let Some(utc) = to_utc(candidate, tz) {
                    if utc <= limit && utc > *after {
                        return Some(utc);
                    }
                }
            }
            ndt = advance_hour_naive(ndt)?;
        }

        None
    }

    fn matches_components(&self, minute: u32, hour: u32, day: u32, month: u32, dow: u32) -> bool {
        if !self.expr.minute.contains(minute) {
            return false;
        }
        if !self.expr.hour.contains(hour) {
            return false;
        }
        if !self.expr.month.contains(month) {
            return false;
        }
        self.day_matches_components(day, dow)
    }

    fn day_matches_components(&self, day: u32, dow: u32) -> bool {
        let dom_restricted = !self.expr.day_of_month.is_all();
        let dow_restricted = !self.expr.day_of_week.is_all();
        let dom_match = self.expr.day_of_month.contains(day);
        let dow_match = self.expr.day_of_week.contains(dow);

        match (dom_restricted, dow_restricted) {
            (true, true) => dom_match || dow_match,
            _ => dom_match && dow_match,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn next_set_value(fs: &crate::parse::FieldSet, from: u32) -> Option<u32> {
    (from..=fs.max_val()).find(|&v| fs.contains(v))
}

/// Convert a naive local datetime to UTC in the given timezone.
/// Returns `None` for DST gaps (times that don't exist).
/// For DST folds (ambiguous times), picks the earliest occurrence.
fn to_utc(ndt: NaiveDateTime, tz: Tz) -> Option<DateTime<Utc>> {
    use chrono::LocalResult;
    match tz.from_local_datetime(&ndt) {
        LocalResult::Single(dt) => Some(dt.with_timezone(&Utc)),
        LocalResult::Ambiguous(earliest, _) => Some(earliest.with_timezone(&Utc)),
        LocalResult::None => None,
    }
}

// --- UTC helpers ---

fn advance_month(dt: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let (year, month) = if dt.month() == 12 {
        (dt.year() + 1, 1)
    } else {
        (dt.year(), dt.month() + 1)
    };
    Utc.with_ymd_and_hms(year, month, 1, 0, 0, 0).single()
}

fn advance_day(dt: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let next = dt + chrono::Duration::days(1);
    Utc.with_ymd_and_hms(next.year(), next.month(), next.day(), 0, 0, 0)
        .single()
}

fn advance_hour(dt: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let next = dt + chrono::Duration::hours(1);
    Utc.with_ymd_and_hms(next.year(), next.month(), next.day(), next.hour(), 0, 0)
        .single()
}

// --- Naive (local time) helpers ---

fn advance_month_naive(dt: NaiveDateTime) -> Option<NaiveDateTime> {
    let (year, month) = if dt.month() == 12 {
        (dt.year() + 1, 1)
    } else {
        (dt.year(), dt.month() + 1)
    };
    NaiveDate::from_ymd_opt(year, month as u32, 1)?.and_hms_opt(0, 0, 0)
}

fn advance_day_naive(dt: NaiveDateTime) -> Option<NaiveDateTime> {
    let next = dt + chrono::Duration::days(1);
    NaiveDate::from_ymd_opt(next.year(), next.month(), next.day())?.and_hms_opt(0, 0, 0)
}

fn advance_hour_naive(dt: NaiveDateTime) -> Option<NaiveDateTime> {
    let next = dt + chrono::Duration::hours(1);
    NaiveDate::from_ymd_opt(next.year(), next.month(), next.day())?.and_hms_opt(next.hour(), 0, 0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn utc(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, mo, d, h, mi, 0).unwrap()
    }

    #[test]
    fn test_matches_every_minute() {
        let sched = CronSchedule::parse("* * * * *").unwrap();
        assert!(sched.matches(&utc(2025, 6, 15, 10, 30)));
    }

    #[test]
    fn test_matches_specific() {
        let sched = CronSchedule::parse("30 14 * * *").unwrap();
        assert!(sched.matches(&utc(2025, 6, 15, 14, 30)));
        assert!(!sched.matches(&utc(2025, 6, 15, 14, 31)));
        assert!(!sched.matches(&utc(2025, 6, 15, 13, 30)));
    }

    #[test]
    fn test_matches_dom_dow_or_logic() {
        // Fire on the 1st OR on Mondays
        let sched = CronSchedule::parse("0 0 1 * MON").unwrap();
        // 2025-06-01 is a Sunday
        assert!(sched.matches(&utc(2025, 6, 1, 0, 0)));
        // 2025-06-02 is a Monday
        assert!(sched.matches(&utc(2025, 6, 2, 0, 0)));
        // 2025-06-03 is a Tuesday and not the 1st
        assert!(!sched.matches(&utc(2025, 6, 3, 0, 0)));
    }

    #[test]
    fn test_next_after_basic() {
        let sched = CronSchedule::parse("30 14 * * *").unwrap();
        let after = utc(2025, 6, 15, 14, 0);
        let next = sched.next_after(&after).unwrap();
        assert_eq!(next, utc(2025, 6, 15, 14, 30));
    }

    #[test]
    fn test_next_after_wraps_day() {
        let sched = CronSchedule::parse("0 9 * * *").unwrap();
        let after = utc(2025, 6, 15, 10, 0);
        let next = sched.next_after(&after).unwrap();
        assert_eq!(next, utc(2025, 6, 16, 9, 0));
    }

    #[test]
    fn test_next_after_wraps_month() {
        let sched = CronSchedule::parse("0 0 1 * *").unwrap();
        let after = utc(2025, 6, 2, 0, 0);
        let next = sched.next_after(&after).unwrap();
        assert_eq!(next, utc(2025, 7, 1, 0, 0));
    }

    #[test]
    fn test_next_after_step() {
        let sched = CronSchedule::parse("*/15 * * * *").unwrap();
        let after = utc(2025, 6, 15, 10, 1);
        let next = sched.next_after(&after).unwrap();
        assert_eq!(next, utc(2025, 6, 15, 10, 15));
    }

    #[test]
    fn test_next_after_returns_none_for_impossible() {
        // Feb 31 can never happen
        let sched = CronSchedule::parse("0 0 31 2 *").unwrap();
        let after = utc(2025, 1, 1, 0, 0);
        assert!(sched.next_after(&after).is_none());
    }

    #[test]
    fn test_next_after_skips_to_same_minute_plus_one() {
        let sched = CronSchedule::parse("30 14 * * *").unwrap();
        // If we're exactly at 14:30, next should be tomorrow 14:30
        let after = utc(2025, 6, 15, 14, 30);
        let next = sched.next_after(&after).unwrap();
        assert_eq!(next, utc(2025, 6, 16, 14, 30));
    }

    // --- Timezone-aware tests ---

    #[test]
    fn test_matches_in_tz_basic() {
        let sched = CronSchedule::parse("0 9 * * *").unwrap();
        let eastern: Tz = "America/New_York".parse().unwrap();
        // 14:00 UTC = 9:00 AM EST (winter)
        assert!(sched.matches_in_tz(&utc(2025, 1, 15, 14, 0), eastern));
        // 13:00 UTC = 9:00 AM EDT (summer)
        assert!(sched.matches_in_tz(&utc(2025, 6, 15, 13, 0), eastern));
        // 14:00 UTC in summer = 10:00 AM EDT, should NOT match
        assert!(!sched.matches_in_tz(&utc(2025, 6, 15, 14, 0), eastern));
    }

    #[test]
    fn test_next_after_in_tz_dst_transition() {
        let sched = CronSchedule::parse("0 9 * * *").unwrap();
        let eastern: Tz = "America/New_York".parse().unwrap();

        // Winter (EST, UTC-5): 9 AM local = 14:00 UTC
        let after_winter = utc(2025, 1, 15, 0, 0);
        let next = sched.next_after_in_tz(&after_winter, eastern).unwrap();
        assert_eq!(next, utc(2025, 1, 15, 14, 0));

        // Summer (EDT, UTC-4): 9 AM local = 13:00 UTC
        let after_summer = utc(2025, 6, 15, 0, 0);
        let next = sched.next_after_in_tz(&after_summer, eastern).unwrap();
        assert_eq!(next, utc(2025, 6, 15, 13, 0));
    }

    #[test]
    fn test_next_after_in_tz_spring_forward_gap() {
        // US spring forward 2025: Mar 9, 2:00 AM -> 3:00 AM
        // Schedule for 2:30 AM — this time doesn't exist on Mar 9
        let sched = CronSchedule::parse("30 2 * * *").unwrap();
        let eastern: Tz = "America/New_York".parse().unwrap();

        let after = utc(2025, 3, 9, 0, 0);
        let next = sched.next_after_in_tz(&after, eastern).unwrap();
        // Should skip Mar 9 (gap) and fire Mar 10 at 2:30 AM EDT = 6:30 UTC
        assert_eq!(next, utc(2025, 3, 10, 6, 30));
    }

    #[test]
    fn test_next_after_in_tz_fall_back_fold() {
        // US fall back 2025: Nov 2, 2:00 AM -> 1:00 AM
        // 1:30 AM occurs twice — should pick the earliest (EDT, UTC-4)
        let sched = CronSchedule::parse("30 1 * * *").unwrap();
        let eastern: Tz = "America/New_York".parse().unwrap();

        let after = utc(2025, 11, 2, 0, 0);
        let next = sched.next_after_in_tz(&after, eastern).unwrap();
        // 1:30 AM EDT = 5:30 UTC (earliest occurrence)
        assert_eq!(next, utc(2025, 11, 2, 5, 30));
    }

    #[test]
    fn test_next_after_in_tz_no_dst_timezone() {
        // UTC+5:30 (India) has no DST
        let sched = CronSchedule::parse("0 9 * * *").unwrap();
        let ist: Tz = "Asia/Kolkata".parse().unwrap();

        let after = utc(2025, 6, 15, 0, 0);
        let next = sched.next_after_in_tz(&after, ist).unwrap();
        // 9:00 AM IST = 3:30 UTC
        assert_eq!(next, utc(2025, 6, 15, 3, 30));
    }

    #[test]
    fn test_next_after_in_tz_wraps_day() {
        let sched = CronSchedule::parse("0 22 * * *").unwrap();
        let eastern: Tz = "America/New_York".parse().unwrap();

        // After 23:00 local (3:00 UTC next day in winter)
        let after = utc(2025, 1, 16, 4, 0); // 11:00 PM EST Jan 15 = past 22:00
        let next = sched.next_after_in_tz(&after, eastern).unwrap();
        // Next 10 PM EST = Jan 17 3:00 UTC
        assert_eq!(next, utc(2025, 1, 17, 3, 0));
    }

    #[test]
    fn test_next_after_in_tz_across_dst_boundary() {
        // Schedule: every day at 9 AM Eastern
        // Check that it correctly transitions from EST to EDT
        let sched = CronSchedule::parse("0 9 * * *").unwrap();
        let eastern: Tz = "America/New_York".parse().unwrap();

        // Mar 8 2025 (still EST): 9 AM = 14:00 UTC
        let after = utc(2025, 3, 8, 14, 0);
        let next = sched.next_after_in_tz(&after, eastern).unwrap();
        // Mar 9 is spring forward day. 9 AM EDT = 13:00 UTC
        assert_eq!(next, utc(2025, 3, 9, 13, 0));
    }

    #[test]
    fn test_next_after_in_tz_negative_offset() {
        // UTC+12 (Auckland, New Zealand) — ahead of UTC
        let sched = CronSchedule::parse("0 8 * * *").unwrap();
        let nzst: Tz = "Pacific/Auckland".parse().unwrap();

        // In July (NZST, UTC+12): 8 AM local = 20:00 UTC previous day
        let after = utc(2025, 7, 14, 21, 0); // 9 AM NZST Jul 15
        let next = sched.next_after_in_tz(&after, nzst).unwrap();
        // 8 AM NZST Jul 16 = 20:00 UTC Jul 15
        assert_eq!(next, utc(2025, 7, 15, 20, 0));
    }

    #[test]
    fn test_next_after_in_tz_step_schedule() {
        // Every 15 minutes in Tokyo (JST, UTC+9, no DST)
        let sched = CronSchedule::parse("*/15 * * * *").unwrap();
        let jst: Tz = "Asia/Tokyo".parse().unwrap();

        let after = utc(2025, 6, 15, 1, 1); // 10:01 AM JST
        let next = sched.next_after_in_tz(&after, jst).unwrap();
        // Next :15 in local = 10:15 JST = 1:15 UTC
        assert_eq!(next, utc(2025, 6, 15, 1, 15));
    }

    #[test]
    fn test_next_after_in_tz_weekday_filter() {
        // Weekdays only at 9 AM Eastern
        let sched = CronSchedule::parse("0 9 * * 1-5").unwrap();
        let eastern: Tz = "America/New_York".parse().unwrap();

        // 2025-06-14 is Saturday
        let after = utc(2025, 6, 14, 0, 0);
        let next = sched.next_after_in_tz(&after, eastern).unwrap();
        // Should skip to Monday Jun 16, 9 AM EDT = 13:00 UTC
        assert_eq!(next, utc(2025, 6, 16, 13, 0));
    }

    #[test]
    fn test_next_after_in_tz_month_boundary() {
        // First of every month at midnight, Berlin time
        let sched = CronSchedule::parse("0 0 1 * *").unwrap();
        let berlin: Tz = "Europe/Berlin".parse().unwrap();

        // After Jun 1 midnight CEST (UTC+2): Jun 1 00:00 CEST = May 31 22:00 UTC
        let after = utc(2025, 5, 31, 22, 0);
        let next = sched.next_after_in_tz(&after, berlin).unwrap();
        // Jul 1 00:00 CEST = Jun 30 22:00 UTC
        assert_eq!(next, utc(2025, 6, 30, 22, 0));
    }

    #[test]
    fn test_matches_in_tz_every_minute() {
        let sched = CronSchedule::parse("* * * * *").unwrap();
        let jst: Tz = "Asia/Tokyo".parse().unwrap();
        assert!(sched.matches_in_tz(&utc(2025, 6, 15, 10, 30), jst));
    }

    #[test]
    fn test_next_after_in_tz_utc_fallback_equivalent() {
        // With UTC timezone, next_after_in_tz should produce the same result as next_after
        let sched = CronSchedule::parse("30 14 * * *").unwrap();
        let utc_tz: Tz = "UTC".parse().unwrap();
        let after = utc(2025, 6, 15, 14, 0);

        let next_utc = sched.next_after(&after).unwrap();
        let next_tz = sched.next_after_in_tz(&after, utc_tz).unwrap();
        assert_eq!(next_utc, next_tz);
    }

    #[test]
    fn test_next_after_in_tz_impossible_schedule() {
        // Feb 31 can never happen — should return None even with timezone
        let sched = CronSchedule::parse("0 0 31 2 *").unwrap();
        let eastern: Tz = "America/New_York".parse().unwrap();
        let after = utc(2025, 1, 1, 0, 0);
        assert!(sched.next_after_in_tz(&after, eastern).is_none());
    }
}
