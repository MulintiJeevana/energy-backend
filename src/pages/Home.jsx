import React from 'react';
import { motion } from 'framer-motion';
import DashboardCards from '../components/DashboardCards';
import ChartsSection from '../components/ChartsSection'
import RecommendationsList from '../components/RecommendationsList';

const Home = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/5">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold bg-gradient-to-r from-energy-primary to-energy-secondary bg-clip-text text-transparent mb-4">
            Smart Energy Dashboard
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Monitor and optimize your home's energy consumption in real-time.
            Get insights, save money, and reduce your carbon footprint.
          </p>
        </motion.div>

        {/* Dashboard Cards */}
        {/* <DashboardCards /> */}

        {/* Charts Section */}
        <ChartsSection />

        {/* Recommendations */}
        <RecommendationsList />

        {/* Quick Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8"
        >
          <div className="text-center p-6 rounded-lg bg-card/50 border border-border/50">
            <div className="text-2xl font-bold text-energy-success mb-2">8</div>
            <div className="text-sm text-muted-foreground">Active Devices</div>
          </div>
          <div className="text-center p-6 rounded-lg bg-card/50 border border-border/50">
            <div className="text-2xl font-bold text-energy-warning mb-2">24%</div>
            <div className="text-sm text-muted-foreground">Peak Hour Usage</div>
          </div>
          {/* <div className="text-center p-6 rounded-lg bg-card/50 border border-border/50">
            <div className="text-2xl font-bold text-energy-primary mb-2">$65</div>
            <div className="text-sm text-muted-foreground">Monthly Savings Potential</div>
          </div> */}
        </motion.div>
      </div>
    </div>
  );
};

export default Home;