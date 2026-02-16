import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { motion } from "framer-motion";
import {
    LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
    XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { apiFetch } from "../lib/api";

const COLORS = ["#22c55e", "#f59e0b", "#ef4444", "#3b82f6", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"];

const ChartsSection = () => {
    const [timeframe, setTimeframe] = useState("hourly");
    const [hourlyData, setHourlyData] = useState([]);
    const [visibleHourly, setVisibleHourly] = useState([]);
    const [dailyData, setDailyData] = useState([]);
    const [appliances, setAppliances] = useState([]);
    const [topAppliances, setTopAppliances] = useState([]);
    const [slotIndex, setSlotIndex] = useState(0);

    const hourlyIntervalRef = useRef(null);
    const slotIntervalRef = useRef(null);

    // ---------- Helpers (memoized) ----------
    const normalizeDevices = useCallback((rawDevices = []) => {
        return rawDevices.map(d => {
            const name = d.name ?? d._id ?? d.device ?? d.device_id ?? "Unknown";
            let slots = null;
            if (Array.isArray(d.slots) && d.slots.length) {
                slots = d.slots.map(s => typeof s === "number" ? s : Number(s.cumulative ?? s.usage ?? s.value ?? 0));
            }
            const dailyUsage = (d.dailyUsage !== undefined) ? Number(d.dailyUsage) : (slots ? Number(slots[slots.length - 1] ?? 0) : (d.currentUsage !== undefined ? Number(d.currentUsage) : 0));
            const currentUsage = (d.currentUsage !== undefined) ? Number(d.currentUsage) : (slots ? Number(slots[slots.length - 1] ?? 0) : dailyUsage);
            return { name, slots, dailyUsage, currentUsage };
        });
    }, []);

    const buildTopAppliancesFromSlot = useCallback((normalizedDevices, idx) => {
        return normalizedDevices.map(d => {
            let usage = 0;
            if (Array.isArray(d.slots) && d.slots.length) {
                usage = d.slots[Math.min(idx, d.slots.length - 1)] ?? 0;
            } else {
                usage = d.dailyUsage ?? d.currentUsage ?? 0;
            }
            return { name: d.name, usage: Number(usage) };
        }).sort((a, b) => b.usage - a.usage);
    }, []);

    // ---------- Stable simulated time (computed once per mount) ----------
    const { currentHour, currentSlot } = useMemo(() => {
        const now = new Date();
        const hour = now.getHours();
        const slot = Math.floor(hour * 3 + now.getMinutes() / 20);
        return { currentHour: hour, currentSlot: slot };
    }, []);

    // ---------- Fetch once ----------
    useEffect(() => {
        let mounted = true;

        async function fetchData() {
            try {
                const now = new Date();
                const dateStr = now.toISOString().slice(0, 10); // "YYYY-MM-DD"
                const month = now.getMonth() + 1;
                const year = now.getFullYear();
                const [hourlyRes, dailyRes, devicesRes] = await Promise.all([
                    apiFetch(`/api/usage/hourly?date=${dateStr}`),
                    apiFetch(`/api/usage/daily?month=${month}&year=${year}`),
                    apiFetch(`/api/usage/devices?date=${dateStr}`)
                ]);

                if (!hourlyRes.ok || !dailyRes.ok || !devicesRes.ok) {
                    console.error("One or more endpoints returned non-OK status");
                }

                const hourly = await hourlyRes.json();
                const daily = await dailyRes.json();
                const devicesRaw = await devicesRes.json();

                if (!mounted) return;

                setHourlyData(Array.isArray(hourly) ? hourly : []);
                setDailyData(Array.isArray(daily) ? daily : []);
                const normalized = normalizeDevices(Array.isArray(devicesRaw) ? devicesRaw : []);
                setAppliances(normalized);

                setSlotIndex(currentSlot);
                setTopAppliances(buildTopAppliancesFromSlot(normalized, currentSlot));
            } catch (err) {
                console.error("Error fetching charts data:", err);
            }
        }

        fetchData();
        return () => { mounted = false; };
    }, [currentSlot, normalizeDevices, buildTopAppliancesFromSlot]);

    // ---------- Animate hourly line up to current hour ----------
    useEffect(() => {
        if (hourlyIntervalRef.current) clearInterval(hourlyIntervalRef.current);
        if (!hourlyData || hourlyData.length === 0) return setVisibleHourly([]);

        setVisibleHourly([hourlyData[0]]);
        let idx = 1;
        hourlyIntervalRef.current = setInterval(() => {
            if (idx <= currentHour) {
                setVisibleHourly(prev => {
                    const last = prev[prev.length - 1];
                    if (last && last.hour === hourlyData[idx]?.hour && last.usage === hourlyData[idx]?.usage) return prev;
                    return [...prev, hourlyData[idx]];
                });
                idx++;
            } else {
                clearInterval(hourlyIntervalRef.current);
            }
        }, 300);
        return () => { if (hourlyIntervalRef.current) clearInterval(hourlyIntervalRef.current); };
    }, [hourlyData, currentHour]);

    // ---------- Animate bar + pie up to currentSlot ----------
    useEffect(() => {
        if (slotIntervalRef.current) clearInterval(slotIntervalRef.current);
        if (!appliances || appliances.length === 0) return;

        let idx = 0;
        const slotsCount = (appliances[0].slots && appliances[0].slots.length) ? appliances[0].slots.length : 72;
        setTopAppliances(buildTopAppliancesFromSlot(appliances, 0));
        setSlotIndex(0);

        slotIntervalRef.current = setInterval(() => {
            if (idx <= currentSlot && idx < slotsCount) {
                setSlotIndex(idx);
                setTopAppliances(buildTopAppliancesFromSlot(appliances, idx));
                idx++;
            } else {
                clearInterval(slotIntervalRef.current);
            }
        }, 400);
        return () => { if (slotIntervalRef.current) clearInterval(slotIntervalRef.current); };
    }, [appliances, currentSlot, buildTopAppliancesFromSlot]);

    // ---------- Memoized derived data ----------
    const chartData = useMemo(
        () => timeframe === "hourly" ? visibleHourly : dailyData,
        [timeframe, visibleHourly, dailyData]
    );

    const pieData = useMemo(
        () => topAppliances.map(d => ({ name: d.name, value: d.usage })),
        [topAppliances]
    );

    const hourlyTicks = useMemo(
        () => (hourlyData && hourlyData.length === 24) ? Array.from({ length: 12 }, (_, i) => i * 2) : undefined,
        [hourlyData]
    );

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {/* Line Chart */}
            <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.6 }}>
                <Card className="h-96">
                    <CardHeader>
                        <div className="flex items-center justify-between">
                            <CardTitle className="text-lg font-semibold">Energy Usage Trends</CardTitle>
                            <div className="flex space-x-2">
                                <Button variant={timeframe === "hourly" ? "default" : "outline"} size="sm" onClick={() => setTimeframe("hourly")} className="text-xs">Hourly</Button>
                                <Button variant={timeframe === "daily" ? "default" : "outline"} size="sm" onClick={() => setTimeframe("daily")} className="text-xs">Daily</Button>
                            </div>
                        </div>
                    </CardHeader>
                    <CardContent>
                        <ResponsiveContainer width="100%" height={240}>
                            <LineChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                                <XAxis dataKey={timeframe === "hourly" ? "hour" : "day"} tick={{ fontSize: 10 }} ticks={timeframe === "hourly" ? hourlyTicks : undefined} />
                                <YAxis tick={{ fontSize: 10 }} />
                                <Tooltip />
                                <Line type="monotone" dataKey="usage" stroke="hsl(var(--energy-primary))" strokeWidth={3} dot={false} activeDot={{ r: 5 }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>
            </motion.div>

            {/* Pie Chart */}
            <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.6, delay: 0.1 }}>
                <Card className="h-96">
                    <CardHeader>
                        <CardTitle className="text-lg font-semibold">Device Usage Breakdown</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <ResponsiveContainer width="100%" height={240}>
                            <PieChart>
                                <Pie data={pieData} cx="50%" cy="50%" outerRadius={80} dataKey="value" nameKey="name" labelLine={false} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                                    {pieData.map((entry, index) => <Cell key={index} fill={COLORS[index % COLORS.length]} />)}
                                </Pie>
                                <Tooltip formatter={value => [`${Number(value).toFixed(2)} kWh`]} />
                            </PieChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>
            </motion.div>

            {/* Bar Chart */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.2 }} className="lg:col-span-2">
                <Card>
                    <CardHeader>
                        <CardTitle className="text-lg font-semibold">Top Consuming Devices (Real-time)</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <ResponsiveContainer width="100%" height={200}>
                            <BarChart data={topAppliances} layout="horizontal">
                                <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                                <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                                <YAxis tick={{ fontSize: 10 }} />
                                <Tooltip formatter={value => [`${Number(value).toFixed(2)} kWh`]} />
                                <Bar dataKey="usage" fill="hsl(var(--energy-secondary))" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>
            </motion.div>
        </div>
    );
};

export default ChartsSection;
