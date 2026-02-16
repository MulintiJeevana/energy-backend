import React, { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { Card, CardContent } from "./ui/hover-card";

const StatCard = React.memo(({ title, value, subtitle, color, delay }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, delay }}
    whileHover={{ y: -4 }}
    className="energy-card-hover"
  >
    <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
      <CardContent>
        <div className="flex items-baseline space-x-2">
          <span className={`text-3xl font-bold ${color}`}>
            {value !== undefined ? value.toFixed(1) : "0.0"}
          </span>
          <span className="text-sm text-muted-foreground">kWh</span>
        </div>
        <div className="mt-2 flex items-center justify-between">
          <span className="text-xs text-muted-foreground">{subtitle}</span>
          <span className={`text-sm font-medium ${color}`}>
            ${(value ? value * 0.12 : 0).toFixed(2)}
          </span>
        </div>
      </CardContent>
    </Card>
  </motion.div>
));

const DashboardCards = () => {
  const [stats, setStats] = useState({ today: 0, week: 0, month: 0 });

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch("/api/usage/stats");
      const data = await res.json();
      setStats(data);
    } catch (err) {
      console.error("❌ Error fetching stats:", err);
    }
  }, []);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 20 * 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchStats]);

  const cards = [
    {
      title: "Today's Usage",
      value: stats.today,
      subtitle: "vs yesterday",
      color: "text-energy-success",
    },
    {
      title: "This Week",
      value: stats.week,
      subtitle: "vs last week",
      color: "text-energy-warning",
    },
    {
      title: "This Month",
      value: stats.month,
      subtitle: "vs last month",
      color: "text-energy-danger",
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      {cards.map((card, index) => (
        <StatCard key={card.title} {...card} delay={index * 0.1} />
      ))}
    </div>
  );
};

export default DashboardCards;
