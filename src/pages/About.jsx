import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Home, Settings, Bell, Moon, BellOff } from 'lucide-react';

const About = () => {
  const technologies = [
    { name: 'React 18', description: 'Modern frontend framework with hooks' },
    // { name: 'TypeScript', description: 'Type-safe JavaScript development' },
    { name: 'Tailwind CSS', description: 'Utility-first CSS framework' },
    { name: 'Framer Motion', description: 'Smooth animations and transitions' },
    { name: 'Recharts', description: 'Responsive chart library' },
    // { name: 'Shadcn/UI', description: 'Beautiful and accessible UI components' },
    { name: 'React Router', description: 'Client-side routing' },
    { name: 'Vite', description: 'Fast build tool and development server' }
  ];

  const features = [
    {
      icon: Home,
      title: 'Real-time Monitoring',
      description: 'Track energy consumption of all your smart appliances in real-time with live updates and instant notifications.'
    },
    {
      icon: Settings,
      title: 'Smart Analytics',
      description: 'Advanced charts and analytics help you understand usage patterns and identify opportunities for savings.'
    },
    {
      icon: Bell,
      title: 'Personalized Recommendations',
      description: 'AI-powered suggestions tailored to your usage patterns help optimize energy consumption and reduce costs.'
    },
    {
      icon: Moon,
      title: 'Dark Mode Support',
      description: 'Beautiful light and dark themes provide a comfortable viewing experience at any time of day.'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/5">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl font-bold bg-gradient-to-r from-energy-primary to-energy-secondary bg-clip-text text-transparent mb-6">
            About EnergyMonitor
          </h1>
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            EnergyMon is a comprehensive smart home energy monitoring dashboard designed to help homeowners
            understand, track, and optimize their energy consumption. Built with modern web technologies,
            it provides real-time insights and actionable recommendations to reduce energy costs and
            environmental impact.
          </p>
        </motion.div>

        {/* Features Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mb-12"
        >
          <h2 className="text-2xl font-bold text-center mb-8">Key Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card className="energy-card-hover h-full border-border/50 bg-card/50 backdrop-blur-sm">
                  <CardHeader>
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-energy-primary/10 rounded-lg flex items-center justify-center">
                        <feature.icon className="w-5 h-5 text-energy-primary" />
                      </div>
                      <CardTitle className="text-lg">{feature.title}</CardTitle>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">{feature.description}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Technologies Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mb-12"
        >
          <h2 className="text-2xl font-bold text-center mb-8">Built With Modern Technologies</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {technologies.map((tech, index) => (
              <motion.div
                key={tech.name}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4, delay: index * 0.05 }}
              >
                <Card className="energy-card-hover h-full text-center p-4 border-border/50 bg-card/50 backdrop-blur-sm">
                  <CardContent className="p-0">
                    <h3 className="font-semibold text-energy-primary mb-2">{tech.name}</h3>
                    <p className="text-sm text-muted-foreground">{tech.description}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Mission Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="mb-12"
        >
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-2xl text-center">Our Mission</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="max-w-3xl mx-auto text-center space-y-4">
                <p className="text-muted-foreground leading-relaxed">
                  We believe that understanding your energy consumption is the first step toward a more
                  sustainable future. Our mission is to make energy monitoring accessible, intuitive,
                  and actionable for every homeowner.
                </p>
                <p className="text-muted-foreground leading-relaxed">
                  By providing clear insights into energy usage patterns and offering personalized
                  recommendations, we empower users to make informed decisions that benefit both
                  their wallets and the environment.
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Stats Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6"
        >
          <Card className="text-center p-6 energy-card-hover">
            <div className="text-3xl font-bold text-energy-success mb-2">25%</div>
            <div className="text-sm text-muted-foreground">Average Energy Savings</div>
          </Card>
          <Card className="text-center p-6 energy-card-hover">
            <div className="text-3xl font-bold text-energy-primary mb-2">24/7</div>
            <div className="text-sm text-muted-foreground">Real-time Monitoring</div>
          </Card>
          <Card className="text-center p-6 energy-card-hover">
            <div className="text-3xl font-bold text-energy-warning mb-2">Smart</div>
            <div className="text-sm text-muted-foreground">AI-Powered Insights</div>
          </Card>
        </motion.div>
      </div>
    </div>
  );
};

export default About;
