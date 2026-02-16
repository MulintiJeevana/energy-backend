import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from './ui/hover-card';
import { Badge } from './ui/badge';
import { Clock, Settings, Bell } from 'lucide-react';
import { recommendations } from '../data/mockData';

const RecommendationsList = () => {
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

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="mb-8"
        >
            <Card className="bg-white dark:bg-gray-800">
                <CardHeader>
                    <CardTitle className="text-lg font-semibold flex items-center space-x-2">
                        <Bell className="w-5 h-5 text-energy-primary" />
                        <span>Energy Saving Recommendations</span>
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="space-y-4">
                        {recommendations.map((rec, index) => {
                            const Icon = getIcon(rec.type);
                            return (
                                <motion.div
                                    key={rec.id}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ duration: 0.4, delay: index * 0.1 }}
                                    className="flex items-start space-x-4 p-4 rounded-lg border border-border/50 bg-card/30 hover:bg-card/80 transition-colors duration-200"
                                >
                                    <div className="flex-shrink-0 w-10 h-10 bg-energy-primary/10 rounded-lg flex items-center justify-center">
                                        <Icon className="w-5 h-5 text-energy-primary" />
                                    </div>

                                    <div className="flex-grow min-w-0">
                                        <div className="flex items-center space-x-2 mb-1">
                                            <h4 className="font-medium text-sm">{rec.appliance}</h4>
                                            <Badge
                                                variant="outline"
                                                className={`text-xs ${getPriorityColor(rec.priority)}`}
                                            >
                                                {rec.priority}
                                            </Badge>
                                        </div>
                                        <p className="text-sm text-muted-foreground mb-2">
                                            {rec.message}
                                        </p>
                                        <div className="flex items-center justify-between">
                                            <span className="text-xs text-muted-foreground capitalize">
                                                {rec.type} optimization
                                            </span>
                                            <span className="text-sm font-medium text-energy-success">
                                                Save {rec.savings}
                                            </span>
                                        </div>
                                    </div>
                                </motion.div>
                            );
                        })}
                    </div>
                </CardContent>
            </Card>
        </motion.div>
    );
};

export default RecommendationsList;
