import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { Clock, Settings, Bell, TrendingDown, Lightbulb } from 'lucide-react';
import { recommendations } from '../data/mockData';

const Recommendations = () => {
    const getIcon = (type) => {
        switch (type) {
            case 'timing':
                return Clock;
            case 'settings':
                return Settings;
            default:
                return Bell;
        }
    };

    const getPriorityColor = (priority) => {
        switch (priority) {
            case 'high':
                return 'bg-energy-danger/10 text-energy-danger border-energy-danger/20';
            case 'medium':
                return 'bg-energy-warning/10 text-energy-warning border-energy-warning/20';
            case 'low':
                return 'bg-energy-success/10 text-energy-success border-energy-success/20';
            default:
                return 'bg-muted text-muted-foreground';
        }
    };

    const totalSavings = recommendations.reduce((sum, rec) => {
        const amount = parseInt(rec.savings.replace(/[^0-9]/g, ''));
        return sum + amount;
    }, 0);

    return (
        <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/5">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    className="mb-8"
                >
                    <h1 className="text-3xl font-bold mb-4">Energy Recommendations</h1>
                    <p className="text-muted-foreground mb-6">
                        Personalized suggestions to optimize your energy consumption and reduce costs
                    </p>

                    {/* Savings Overview */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                        <Card className="energy-card-hover">
                            <CardContent className="p-6 text-center">
                                <TrendingDown className="w-8 h-8 text-energy-success mx-auto mb-2" />
                                <div className="text-2xl font-bold text-energy-success mb-1">
                                    ${totalSavings}
                                </div>
                                <div className="text-sm text-muted-foreground">
                                    Potential Monthly Savings
                                </div>
                            </CardContent>
                        </Card>

                        <Card className="energy-card-hover">
                            <CardContent className="p-6 text-center">
                                <Lightbulb className="w-8 h-8 text-energy-warning mx-auto mb-2" />
                                <div className="text-2xl font-bold text-energy-warning mb-1">
                                    {recommendations.length}
                                </div>
                                <div className="text-sm text-muted-foreground">
                                    Active Recommendations
                                </div>
                            </CardContent>
                        </Card>

                        <Card className="energy-card-hover">
                            <CardContent className="p-6 text-center">
                                <Settings className="w-8 h-8 text-energy-primary mx-auto mb-2" />
                                <div className="text-2xl font-bold text-energy-primary mb-1">
                                    {recommendations.filter(r => r.priority === 'high').length}
                                </div>
                                <div className="text-sm text-muted-foreground">
                                    High Priority Actions
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </motion.div>

                {/* Recommendations List */}
                <div className="space-y-6">
                    {recommendations.map((rec, index) => {
                        const Icon = getIcon(rec.type);
                        return (
                            <motion.div
                                key={rec.id}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ duration: 0.5, delay: index * 0.1 }}
                            >
                                <Card className="energy-card-hover border-border/50 bg-card/50 backdrop-blur-sm">
                                    <CardHeader>
                                        <div className="flex items-start justify-between">
                                            <div className="flex items-center space-x-4">
                                                <div className="flex-shrink-0 w-12 h-12 bg-energy-primary/10 rounded-lg flex items-center justify-center">
                                                    <Icon className="w-6 h-6 text-energy-primary" />
                                                </div>
                                                <div>
                                                    <CardTitle className="text-xl font-semibold mb-1">
                                                        {rec.appliance}
                                                    </CardTitle>
                                                    <div className="flex items-center space-x-2">
                                                        <Badge
                                                            variant="outline"
                                                            className={`text-xs ${getPriorityColor(rec.priority)}`}
                                                        >
                                                            {rec.priority} priority
                                                        </Badge>
                                                        <span className="text-sm text-muted-foreground capitalize">
                                                            {rec.type} optimization
                                                        </span>
                                                    </div>
                                                </div>
                                            </div>
                                            <div className="text-right">
                                                <div className="text-lg font-bold text-energy-success">
                                                    {rec.savings}
                                                </div>
                                                <div className="text-xs text-muted-foreground">
                                                    potential savings
                                                </div>
                                            </div>
                                        </div>
                                    </CardHeader>

                                    <CardContent>
                                        <p className="text-muted-foreground mb-4">
                                            {rec.message}
                                        </p>

                                        <div className="flex items-center justify-between">
                                            <div className="flex space-x-2">
                                                <Button size="sm" className="bg-energy-primary hover:bg-energy-primary/90">
                                                    Apply Now
                                                </Button>
                                                <Button variant="outline" size="sm">
                                                    Learn More
                                                </Button>
                                            </div>
                                            <span className="text-xs text-muted-foreground">
                                                Est. implementation: 2 min
                                            </span>
                                        </div>
                                    </CardContent>
                                </Card>
                            </motion.div>
                        );
                    })}
                </div>

                {/* Energy Saving Tips */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.5 }}
                    className="mt-12"
                >
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-xl font-semibold flex items-center space-x-2">
                                <Lightbulb className="w-5 h-5 text-energy-warning" />
                                <span>General Energy Saving Tips</span>
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="space-y-4">
                                    <h4 className="font-medium text-foreground">Quick Wins</h4>
                                    <ul className="space-y-2 text-sm text-muted-foreground">
                                        <li>• Replace incandescent bulbs with LED alternatives</li>
                                        <li>• Unplug devices when not in use to eliminate phantom loads</li>
                                        <li>• Use power strips to easily cut power to multiple devices</li>
                                        <li>• Adjust thermostat by 2-3°F to reduce HVAC costs</li>
                                    </ul>
                                </div>
                                <div className="space-y-4">
                                    <h4 className="font-medium text-foreground">Long-term Strategies</h4>
                                    <ul className="space-y-2 text-sm text-muted-foreground">
                                        <li>• Upgrade to ENERGY STAR certified appliances</li>
                                        <li>• Install programmable or smart thermostats</li>
                                        <li>• Consider solar panels for renewable energy</li>
                                        <li>• Improve home insulation to reduce heating/cooling needs</li>
                                    </ul>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </motion.div>
            </div>
        </div>
    );
};

export default Recommendations;
