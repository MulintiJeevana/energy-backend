import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, PieChart, Pie, Cell, Legend,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Zap, Calendar, TrendingUp, BarChart3 } from 'lucide-react';
import { apiFetch } from '../lib/api';

const COLORS = [
    '#3B82F6', '#8B5CF6', '#10B981', '#F59E0B',
    '#EF4444', '#EC4899', '#14B8A6', '#F97316',
];

/** Format a Date to YYYY-MM-DD for <input type="date"> */
const toInputDate = (d) => d.toISOString().split('T')[0];

/** Validate YYYY-MM-DD string */
const isValidDate = (str) =>
    typeof str === 'string' &&
    /^\d{4}-\d{2}-\d{2}$/.test(str) &&
    !isNaN(new Date(str + 'T00:00:00').getTime());

/**
 * Fill missing dates in the daily-usage array with { date, usage: 0 }.
 * Produces a continuous day-by-day timeline from startDate to endDate.
 */
const fillDateGaps = (data, startDate, endDate) => {
    if (!isValidDate(startDate) || !isValidDate(endDate)) return data;

    const lookup = new Map(data.map((d) => [d.date, d.usage]));
    const filled = [];
    const cur = new Date(startDate + 'T00:00:00');
    const end = new Date(endDate + 'T00:00:00');

    while (cur <= end) {
        const key = cur.toISOString().split('T')[0];
        filled.push({ date: key, usage: lookup.get(key) ?? 0 });
        cur.setDate(cur.getDate() + 1);
    }
    return filled;
};

/**
 * Adaptive X-axis label formatter.
 * - ≤ 30 days  → "Mar 3"
 * - ≤ 365 days → "Mar 2024"
 * - > 365 days → "Mar '24"
 */
const makeTickFormatter = (daySpan) => (iso) => {
    const d = new Date(iso + 'T00:00:00');
    if (daySpan <= 30) {
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }
    if (daySpan <= 365) {
        return d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
    }
    return d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
};

/** Compute how many ticks to skip so the axis stays readable */
const computeTickInterval = (dataLength) => {
    if (dataLength <= 10) return 0;         // show every label
    if (dataLength <= 20) return 1;         // every other
    if (dataLength <= 90) return Math.ceil(dataLength / 8) - 1;
    if (dataLength <= 365) return Math.ceil(dataLength / 6) - 1;
    return Math.ceil(dataLength / 5) - 1;   // multi-year
};

/** Tooltip always shows full date context */
const tooltipDate = (iso) => {
    const d = new Date(iso + 'T00:00:00');
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
};

/* ─── Pie label renderer ─── */
const renderPieLabel = ({ name, percent }) =>
    `${name} (${(percent * 100).toFixed(0)}%)`;

/* ─── Stat Card ─── */
const StatCard = React.memo(({ icon: Icon, label, value, unit, color, delay }) => (
    <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay }}
    >
        <Card className="energy-card-hover border-border/50 bg-card/50 backdrop-blur-sm">
            <CardContent className="flex items-center p-5 gap-4">
                <div className={`p-3 rounded-xl ${color}`}>
                    <Icon className="w-5 h-5 text-white" />
                </div>
                <div>
                    <p className="text-sm text-muted-foreground">{label}</p>
                    <p className="text-2xl font-bold">
                        {value}
                        <span className="text-sm font-normal text-muted-foreground ml-1">{unit}</span>
                    </p>
                </div>
            </CardContent>
        </Card>
    </motion.div>
));

/* ─── Custom Recharts Tooltip ─── */
const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    return (
        <div className="bg-card border border-border rounded-lg shadow-lg px-4 py-2">
            <p className="text-sm font-medium text-foreground">{tooltipDate(label)}</p>
            <p className="text-sm text-energy-primary font-semibold">
                {payload[0].value.toFixed(2)} kWh
            </p>
        </div>
    );
};

const PieTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    return (
        <div className="bg-card border border-border rounded-lg shadow-lg px-4 py-2">
            <p className="text-sm font-medium text-foreground">{payload[0].name}</p>
            <p className="text-sm font-semibold" style={{ color: payload[0].payload.fill }}>
                {payload[0].value.toFixed(2)} kWh
            </p>
        </div>
    );
};

/* ─── Main Page ─── */
const Analytics = () => {
    const today = useMemo(() => new Date(), []);
    const weekAgo = useMemo(() => {
        const d = new Date(today);
        d.setDate(d.getDate() - 6);
        return d;
    }, [today]);

    const todayStr = useMemo(() => toInputDate(today), [today]);

    const [startDate, setStartDate] = useState(toInputDate(weekAgo));
    const [endDate, setEndDate] = useState(todayStr);
    const [dateError, setDateError] = useState(null);
    const [dailyData, setDailyData] = useState([]);
    const [deviceData, setDeviceData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    /** Validate a pair of date strings and return an error message or null */
    const validateRange = (start, end) => {
        if (!isValidDate(start)) return 'Invalid start date';
        if (!isValidDate(end)) return 'Invalid end date';
        if (start > end) return 'Start date cannot be after end date';
        if (end > todayStr) return 'End date cannot be in the future';
        return null;
    };

    /** onInput: update state immediately so the field is editable (keyboard) */
    const onStartInput = (e) => setStartDate(e.target.value);
    const onEndInput = (e) => setEndDate(e.target.value);

    /** onChange: fires on committed value (calendar pick or completed keyboard entry) */
    const onStartChange = (e) => {
        const val = e.target.value;
        setStartDate(val);
        setDateError(validateRange(val, endDate));
    };
    const onEndChange = (e) => {
        const val = e.target.value;
        setEndDate(val);
        setDateError(validateRange(startDate, val));
    };

    /** onBlur: validate when user leaves the field (catches partial keyboard input) */
    const onStartBlur = () => setDateError(validateRange(startDate, endDate));
    const onEndBlur = () => setDateError(validateRange(startDate, endDate));

    const fetchAnalytics = useCallback(async () => {
        // Skip fetch if dates are invalid or inverted
        if (!isValidDate(startDate) || !isValidDate(endDate) || startDate > endDate) return;

        setLoading(true);
        setError(null);
        try {
            const [dailyRes, deviceRes] = await Promise.all([
                apiFetch(`/api/analytics/daily-usage?startDate=${startDate}&endDate=${endDate}`),
                apiFetch(`/api/analytics/device-usage?startDate=${startDate}&endDate=${endDate}`),
            ]);

            if (!dailyRes.ok || !deviceRes.ok) throw new Error('Failed to fetch analytics');

            const dailyJson = await dailyRes.json();
            const deviceJson = await deviceRes.json();

            setDailyData(dailyJson.data || []);  // raw API data; gap-fill later via useMemo
            setDeviceData(
                (deviceJson.data || []).map((d, i) => ({
                    ...d,
                    name: d.device,
                    fill: COLORS[i % COLORS.length],
                }))
            );
        } catch (err) {
            console.error('Analytics fetch error:', err);
            setError('Unable to load analytics. Please try again.');
        } finally {
            setLoading(false);
        }
    }, [startDate, endDate]);

    useEffect(() => {
        fetchAnalytics();
    }, [fetchAnalytics]);

    /* Compute day span and adaptive chart config */
    const daySpan = useMemo(() => {
        if (!isValidDate(startDate) || !isValidDate(endDate)) return 7;
        const ms = new Date(endDate).getTime() - new Date(startDate).getTime();
        return Math.max(1, Math.ceil(ms / (1000 * 60 * 60 * 24)));
    }, [startDate, endDate]);

    const tickFormatter = useMemo(() => makeTickFormatter(daySpan), [daySpan]);

    /* ─── ISSUE 4 FIX: Fill gaps in daily data with 0-usage entries ─── */
    const filledDailyData = useMemo(
        () => fillDateGaps(dailyData, startDate, endDate),
        [dailyData, startDate, endDate]
    );

    const tickInterval = useMemo(() => computeTickInterval(filledDailyData.length), [filledDailyData.length]);
    const tickAngle = useMemo(() => (filledDailyData.length > 15 ? -35 : 0), [filledDailyData.length]);

    /* Derived summary stats — use filledDailyData for consistent day count */
    const totalKwh = useMemo(
        () => filledDailyData.reduce((sum, d) => sum + d.usage, 0),
        [filledDailyData]
    );
    const avgDaily = useMemo(
        () => (filledDailyData.length ? totalKwh / filledDailyData.length : 0),
        [totalKwh, filledDailyData.length]
    );
    const estimatedCost = useMemo(() => totalKwh * 0.12, [totalKwh]);

    /* Quick range buttons */
    const setRange = (days) => {
        const end = new Date();
        const start = new Date();
        start.setDate(end.getDate() - (days - 1));
        setStartDate(toInputDate(start));
        setEndDate(toInputDate(end));
        setDateError(null);
    };

    /* ─── Render ─── */
    return (
        <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/5">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    className="mb-8"
                >
                    <h1 className="text-3xl font-bold mb-2">Energy Analytics</h1>
                    <p className="text-muted-foreground">
                        Analyze your energy consumption patterns over custom date ranges
                    </p>
                </motion.div>

                {/* Date Range Controls */}
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.1 }}
                >
                    <Card className="mb-8 border-border/50 bg-card/50 backdrop-blur-sm">
                        <CardContent className="p-5">
                            <div className="flex flex-wrap items-end gap-4">
                                <div className="flex items-center gap-2">
                                    <Calendar className="w-4 h-4 text-muted-foreground" />
                                    <span className="text-sm font-medium text-muted-foreground">Date Range</span>
                                </div>

                                <div className="flex flex-col gap-1">
                                    <div className="flex items-center gap-2">
                                        <input
                                            id="analytics-start-date"
                                            type="date"
                                            value={startDate}
                                            onInput={onStartInput}
                                            onChange={onStartChange}
                                            onBlur={onStartBlur}
                                            className={`px-3 py-1.5 rounded-md border bg-background text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-energy-primary/50 ${dateError ? 'border-red-500' : 'border-border'}`}
                                        />
                                        <span className="text-muted-foreground">to</span>
                                        <input
                                            id="analytics-end-date"
                                            type="date"
                                            value={endDate}
                                            onInput={onEndInput}
                                            onChange={onEndChange}
                                            onBlur={onEndBlur}
                                            max={todayStr}
                                            className={`px-3 py-1.5 rounded-md border bg-background text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-energy-primary/50 ${dateError ? 'border-red-500' : 'border-border'}`}
                                        />
                                    </div>
                                    {dateError && (
                                        <span className="text-xs text-red-500">{dateError}</span>
                                    )}
                                </div>

                                {/* Quick range buttons */}
                                <div className="flex gap-2">
                                    {[
                                        { label: '7D', days: 7 },
                                        { label: '14D', days: 14 },
                                        { label: '30D', days: 30 },
                                        { label: '90D', days: 90 },
                                    ].map((r) => (
                                        <Button
                                            key={r.label}
                                            variant="outline"
                                            size="sm"
                                            onClick={() => setRange(r.days)}
                                            className="text-xs"
                                        >
                                            {r.label}
                                        </Button>
                                    ))}
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </motion.div>

                {/* Error State */}
                {error && (
                    <Card className="mb-8 border-red-500/30 bg-red-500/5">
                        <CardContent className="p-5 text-center text-red-500 text-sm">
                            {error}
                            <Button variant="outline" size="sm" className="ml-3" onClick={fetchAnalytics}>
                                Retry
                            </Button>
                        </CardContent>
                    </Card>
                )}

                {/* Loading State */}
                {loading && (
                    <div className="flex items-center justify-center py-20">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-energy-primary" />
                    </div>
                )}

                {/* Content */}
                {!loading && !error && (
                    <>
                        {/* Summary Cards */}
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
                            <StatCard
                                icon={Zap}
                                label="Total Consumption"
                                value={totalKwh.toFixed(2)}
                                unit="kWh"
                                color="bg-blue-500"
                                delay={0}
                            />
                            <StatCard
                                icon={TrendingUp}
                                label="Avg. Daily Usage"
                                value={avgDaily.toFixed(2)}
                                unit="kWh"
                                color="bg-purple-500"
                                delay={0.1}
                            />
                            <StatCard
                                icon={BarChart3}
                                label="Estimated Cost"
                                value={`$${estimatedCost.toFixed(2)}`}
                                unit=""
                                color="bg-emerald-500"
                                delay={0.2}
                            />
                        </div>

                        {/* Charts Row */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            {/* Line Chart — spans 2 cols */}
                            <motion.div
                                className="lg:col-span-2"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.6, delay: 0.2 }}
                            >
                                <Card className="h-full border-border/50 bg-card/50 backdrop-blur-sm">
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-lg font-semibold">Daily Consumption</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        {filledDailyData.length === 0 ? (
                                            <div className="flex items-center justify-center h-64 text-muted-foreground text-sm">
                                                No data for the selected date range
                                            </div>
                                        ) : (
                                            <ResponsiveContainer width="100%" height={320}>
                                                <LineChart data={filledDailyData} margin={{ top: 5, right: 10, bottom: tickAngle ? 20 : 5, left: 0 }}>
                                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                                                    <XAxis
                                                        dataKey="date"
                                                        tickFormatter={tickFormatter}
                                                        interval={tickInterval}
                                                        angle={tickAngle}
                                                        textAnchor={tickAngle ? 'end' : 'middle'}
                                                        height={tickAngle ? 60 : 30}
                                                        tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
                                                        padding={{ left: 0, right: 0 }}
                                                    />
                                                    <YAxis
                                                        tick={{ fontSize: 12, fill: 'var(--muted-foreground)' }}
                                                        tickFormatter={(v) => `${v}`}
                                                        unit=" kWh"
                                                    />
                                                    <Tooltip content={<CustomTooltip />} />
                                                    <Line
                                                        type="monotone"
                                                        dataKey="usage"
                                                        stroke="#3B82F6"
                                                        strokeWidth={2.5}
                                                        dot={filledDailyData.length <= 60 ? { r: 3, fill: '#3B82F6' } : false}
                                                        activeDot={{ r: 6 }}
                                                    />
                                                </LineChart>
                                            </ResponsiveContainer>
                                        )}
                                    </CardContent>
                                </Card>
                            </motion.div>

                            {/* Pie Chart — 1 col */}
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.6, delay: 0.3 }}
                            >
                                <Card className="h-full border-border/50 bg-card/50 backdrop-blur-sm">
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-lg font-semibold">Device Breakdown</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        {deviceData.length === 0 ? (
                                            <div className="flex items-center justify-center h-64 text-muted-foreground text-sm">
                                                No device data available
                                            </div>
                                        ) : (
                                            <ResponsiveContainer width="100%" height={320}>
                                                <PieChart>
                                                    <Pie
                                                        data={deviceData}
                                                        dataKey="usage"
                                                        nameKey="name"
                                                        cx="50%"
                                                        cy="45%"
                                                        outerRadius={90}
                                                        label={renderPieLabel}
                                                        labelLine={{ stroke: 'var(--muted-foreground)' }}
                                                    >
                                                        {deviceData.map((entry, i) => (
                                                            <Cell key={i} fill={entry.fill} />
                                                        ))}
                                                    </Pie>
                                                    <Tooltip content={<PieTooltip />} />
                                                    <Legend
                                                        verticalAlign="bottom"
                                                        iconType="circle"
                                                        iconSize={8}
                                                        wrapperStyle={{ fontSize: 12 }}
                                                    />
                                                </PieChart>
                                            </ResponsiveContainer>
                                        )}
                                    </CardContent>
                                </Card>
                            </motion.div>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default Analytics;
