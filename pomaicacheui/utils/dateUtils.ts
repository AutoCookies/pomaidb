export function parseDbTimestamp(ts?: string | null, createdAt?: string | null): Date | null {
    if (!ts || ts === "NULL") return createdAt ? safeDate(createdAt) : null;

    // Try native Date parse first (ISO / RFC3339 / timestamptz)
    const d = safeDate(ts);
    if (d) return d;

    // Match time-only formats like "22:37:31.223007+07" or "22:37:31.223007"
    const timeOnly = /^(\d{2}):(\d{2}):(\d{2})(\.\d+)?(?:([+-]\d{2})(?::?(\d{2}))?)?$/;
    const m = ts.match(timeOnly);
    if (m) {
        const hour = Number(m[1]);
        const minute = Number(m[2]);
        const second = Number(m[3]);
        const fraction = m[4] ? Number("0" + m[4]) : 0; // fractional seconds as 0.xxx
        // Use createdAt's date if available, otherwise use today's date
        const base = createdAt ? safeDate(createdAt) : new Date();
        if (!base) return null;
        const out = new Date(base);
        const milli = Math.round(fraction * 1000);
        out.setHours(hour, minute, second, milli);

        // If there's a timezone offset in the time string (e.g. +07), we attempt to adjust:
        if (m[5]) {
            const tzHour = Number(m[5]);
            const tzMin = m[6] ? Number(m[6]) : 0;
            // Convert offset to minutes
            const offsetMinutes = tzHour * 60 + tzMin;
            // JavaScript Date stores time in local timezone; apply offset difference to normalize to UTC-like moment.
            // We'll subtract offset to get the equivalent UTC time (approximate, good-enough for display).
            out.setTime(out.getTime() - offsetMinutes * 60 * 1000);
        }

        return out;
    }

    // Could not parse
    return null;
}

export function formatDbTimestamp(ts?: string | null, createdAt?: string | null, locale = undefined): string {
    const d = parseDbTimestamp(ts, createdAt);
    if (!d) return "—";
    try {
        // default locale-based format, show date + time
        return d.toLocaleString(locale);
    } catch {
        return d.toISOString();
    }
}

// Helper: try to parse string into Date and ensure valid
function safeDate(s: string): Date | null {
    const d = new Date(s);
    return isNaN(d.getTime()) ? null : d;
}