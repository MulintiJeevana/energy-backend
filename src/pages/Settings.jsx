import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
//add
import { Switch } from '../components/ui/Switch';
import { useTheme } from '../context/ThemeContext';
import { Bell, Settings as SettingsIcon, Home } from 'lucide-react';

const Settings = () => {
    const { theme, toggleTheme } = useTheme();
    const [notifications, setNotifications] = useState(true);
    const [autoOptimize, setAutoOptimize] = useState(false);
    const [dataCollection, setDataCollection] = useState(true);

    const handleSaveSettings = () => {
        console.log('Settings saved:', {
            theme,
            notifications,
            autoOptimize,
            dataCollection
        });
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/5">
            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    className="mb-8"
                >
                    <h1 className="text-3xl font-bold mb-4">Settings</h1>
                    <p className="text-muted-foreground">
                        Customize your EnergyMon experience and preferences
                    </p>
                </motion.div>

                {/* Appearance Settings */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.1 }}
                    className="mb-6"
                >
                    <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                        <CardHeader>
                            <CardTitle className="flex items-center space-x-2">
                                <SettingsIcon className="w-5 h-5 text-energy-primary" />
                                <span>Appearance</span>
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <h3 className="font-medium">Dark Mode</h3>
                                    <p className="text-sm text-muted-foreground">
                                        Toggle between light and dark themes
                                    </p>
                                </div>
                                <Switch
                                    checked={theme === 'dark'}
                                    onCheckedChange={toggleTheme}
                                />
                            </div>
                        </CardContent>
                    </Card>
                </motion.div>

                {/* Notification Settings */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                    className="mb-6"
                >
                    <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                        <CardHeader>
                            <CardTitle className="flex items-center space-x-2">
                                <Bell className="w-5 h-5 text-energy-primary" />
                                <span>Notifications</span>
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <h3 className="font-medium">Energy Alerts</h3>
                                    <p className="text-sm text-muted-foreground">
                                        Get notified about high energy usage and recommendations
                                    </p>
                                </div>
                                <Switch
                                    checked={notifications}
                                    onCheckedChange={setNotifications}
                                />
                            </div>

                            <div className="flex items-center justify-between">
                                <div>
                                    <h3 className="font-medium">Auto-Optimization</h3>
                                    <p className="text-sm text-muted-foreground">
                                        Automatically apply energy-saving recommendations
                                    </p>
                                </div>
                                <Switch
                                    checked={autoOptimize}
                                    onCheckedChange={setAutoOptimize}
                                />
                            </div>
                        </CardContent>
                    </Card>
                </motion.div>

                {/* Privacy Settings */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.3 }}
                    className="mb-6"
                >
                    <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                        <CardHeader>
                            <CardTitle className="flex items-center space-x-2">
                                <Home className="w-5 h-5 text-energy-primary" />
                                <span>Privacy & Data</span>
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <h3 className="font-medium">Data Collection</h3>
                                    <p className="text-sm text-muted-foreground">
                                        Allow collection of usage data to improve recommendations
                                    </p>
                                </div>
                                <Switch
                                    checked={dataCollection}
                                    onCheckedChange={setDataCollection}
                                />
                            </div>
                        </CardContent>
                    </Card>
                </motion.div>

                {/* Account Settings */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.4 }}
                    className="mb-8"
                >
                    <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                        <CardHeader>
                            <CardTitle>Account Information</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <h3 className="font-medium mb-1">Home Address</h3>
                                    <p className="text-sm text-muted-foreground">123 Smart Home Lane</p>
                                </div>
                                <div>
                                    <h3 className="font-medium mb-1">Utility Provider</h3>
                                    <p className="text-sm text-muted-foreground">Green Energy Co.</p>
                                </div>
                                <div>
                                    <h3 className="font-medium mb-1">Rate Plan</h3>
                                    <p className="text-sm text-muted-foreground">Time-of-Use</p>
                                </div>
                                <div>
                                    <h3 className="font-medium mb-1">Monthly Budget</h3>
                                    <p className="text-sm text-muted-foreground">$150.00</p>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </motion.div>

                {/* Save Button */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.5 }}
                    className="flex justify-end space-x-4"
                >
                    <Button variant="outline">Reset to Defaults</Button>
                    <Button
                        onClick={handleSaveSettings}
                        className="bg-energy-primary hover:bg-energy-primary/90"
                    >
                        Save Settings
                    </Button>
                </motion.div>

                {/* Settings Summary */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.6 }}
                    className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4"
                >
                    <Card className="text-center p-4">
                        <div className="text-lg font-semibold text-energy-primary mb-1">
                            {theme === 'dark' ? 'Dark' : 'Light'}
                        </div>
                        <div className="text-xs text-muted-foreground">Current Theme</div>
                    </Card>
                    <Card className="text-center p-4">
                        <div className="text-lg font-semibold text-energy-success mb-1">
                            {notifications ? 'On' : 'Off'}
                        </div>
                        <div className="text-xs text-muted-foreground">Notifications</div>
                    </Card>
                    <Card className="text-center p-4">
                        <div className="text-lg font-semibold text-energy-warning mb-1">
                            {autoOptimize ? 'On' : 'Off'}
                        </div>
                        <div className="text-xs text-muted-foreground">Auto-Optimize</div>
                    </Card>
                </motion.div>
            </div>
        </div>
    );
};

export default Settings;
