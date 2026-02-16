import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { apiFetch } from '../lib/api';

const Appliances = () => {
    const [filter, setFilter] = useState('all');
    const [appliances, setAppliances] = useState([]); // State for fetched data
    const [loading, setLoading] = useState(true); // Optional: loading state
    useEffect(() => {
        const today = new Date().toISOString().slice(0, 10);
        apiFetch(`/api/devices?date=${today}`)
            .then(res => res.json())
            .then(data => {
                setAppliances(Array.isArray(data) ? data : []);
                // setAppliances(data);
                setLoading(false);
            })
            .catch(err => {
                console.error(err);
                setLoading(false);
            });
        console.log("apliances", appliances)
    }, []);

    const filteredAppliances = appliances.filter((appliance) => {
        if (filter === 'all') return true;
        return appliance.status === filter;
    });

    const getStatusColor = (status) => {
        switch (status) {
            case 'on':
                return 'bg-energy-success/10 text-energy-success border-energy-success/20';
            case 'off':
                return 'bg-muted/50 text-muted-foreground border-border';
            case 'standby':
                return 'bg-energy-warning/10 text-energy-warning border-energy-warning/20';
            default:
                return 'bg-muted text-muted-foreground';
        }
    };

    const getEfficiencyColor = (efficiency) => {
        switch (efficiency) {
            case 'high':
                return 'text-energy-success';
            case 'medium':
                return 'text-energy-warning';
            case 'low':
                return 'text-energy-danger';
            default:
                return 'text-muted-foreground';
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <span className="text-muted-foreground text-lg">Loading devices...</span>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/5">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    className="mb-8"
                >
                    <h1 className="text-3xl font-bold mb-4">Smart Appliances</h1>
                    <p className="text-muted-foreground mb-6">
                        Monitor your home appliances in real-time
                    </p>

                    {/* Filter Buttons */}
                    <div className="flex flex-wrap gap-2 mb-6">
                        {['all', 'on', 'off', 'standby'].map((status) => (
                            <Button
                                key={status}
                                variant={filter === status ? 'default' : 'outline'}
                                size="sm"
                                onClick={() => setFilter(status)}
                                className="capitalize"
                            >
                                {status === 'all' ? 'All Devices' : status}
                            </Button>
                        ))}
                    </div>
                </motion.div>

                {/* Appliances Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {filteredAppliances.map((appliance, index) => (
                        <motion.div
                            key={appliance.id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5, delay: index * 0.1 }}
                        >
                            <Card className="energy-card-hover border-border/50 bg-card/50 backdrop-blur-sm">
                                <CardHeader className="pb-4">
                                    <div className="flex items-center justify-between">
                                        <CardTitle className="text-lg font-semibold">
                                            {appliance.name}
                                        </CardTitle>
                                        <Badge
                                            variant="outline"
                                            className={getStatusColor(appliance.status)}
                                        >
                                            {appliance.status}
                                        </Badge>
                                    </div>
                                    <div className="text-sm text-muted-foreground">
                                        {appliance.category} • {appliance.location}
                                    </div>
                                </CardHeader>

                                <CardContent>
                                    {/* Current Usage */}
                                    <div className="mb-4">
                                        <div className="flex items-baseline space-x-2 mb-1">
                                            <span className="text-2xl font-bold text-energy-primary">
                                                {appliance.currentUsage > 0
                                                    ? (appliance.currentUsage).toFixed(3)
                                                    : '0.000'}
                                            </span>
                                            <span className="text-sm text-muted-foreground">kW</span>
                                        </div>
                                        <div className="text-xs text-muted-foreground">Current Usage</div>
                                    </div>

                                    {/* Usage Stats */}
                                    <div className="grid grid-cols-3 gap-4 mb-4">
                                        <div className="text-center">
                                            <div className="text-sm font-medium text-foreground">
                                                {appliance.dailyUsage.toFixed(1)}
                                            </div>
                                            <div className="text-xs text-muted-foreground">Daily kWh</div>
                                        </div>
                                        <div className="text-center">
                                            <div className="text-sm font-medium text-foreground">
                                                {appliance.weeklyUsage.toFixed(1)}
                                            </div>
                                            <div className="text-xs text-muted-foreground">Weekly kWh</div>
                                        </div>
                                        <div className="text-center">
                                            <div className="text-sm font-medium text-foreground">
                                                {appliance.monthlyUsage}
                                            </div>
                                            <div className="text-xs text-muted-foreground">Monthly kWh</div>
                                        </div>
                                    </div>

                                    {/* Efficiency Rating */}
                                    <div className="flex items-center justify-between">
                                        <span className="text-sm text-muted-foreground">Efficiency</span>
                                        <span
                                            className={`text-sm font-medium capitalize ${getEfficiencyColor(
                                                appliance.efficiency
                                            )}`}
                                        >
                                            {appliance.efficiency}
                                        </span>
                                    </div>
                                </CardContent>
                            </Card>
                        </motion.div>
                    ))}
                </div>

                {/* Summary Stats */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.3 }}
                    className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-4"
                >
                    <Card className="text-center p-4">
                        <div className="text-2xl font-bold text-energy-success mb-1">
                            {appliances.filter((a) => a.status === 'on').length}
                        </div>
                        <div className="text-sm text-muted-foreground">Active</div>
                    </Card>
                    <Card className="text-center p-4">
                        <div className="text-2xl font-bold text-energy-warning mb-1">
                            {appliances.filter((a) => a.status === 'standby').length}
                        </div>
                        <div className="text-sm text-muted-foreground">Standby</div>
                    </Card>
                    <Card className="text-center p-4">
                        <div className="text-2xl font-bold text-muted-foreground mb-1">
                            {appliances.filter((a) => a.status === 'off').length}
                        </div>
                        <div className="text-sm text-muted-foreground">Off</div>
                    </Card>
                    <Card className="text-center p-4">
                        <div className="text-2xl font-bold text-energy-primary mb-1">
                            {(appliances.reduce((sum, a) => sum + a.currentUsage, 0) / 1000).toFixed(1)}
                        </div>
                        <div className="text-sm text-muted-foreground">Total kW</div>
                    </Card>
                </motion.div>
            </div>
        </div>
    );
};

export default Appliances;
