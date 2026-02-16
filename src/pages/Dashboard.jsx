import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line
} from 'recharts';
import {
  Zap, TrendingUp, TrendingDown,
  Lightbulb
} from 'lucide-react';
import { apiFetch } from '../lib/api';

const COLORS = ['#3B82F6', '#8B5CF6', '#10B981', '#F59E0B', '#EF4444', '#6B7280'];

/** Reusable chart card wrapper to reduce JSX repetition */
const ChartCard = React.memo(({ title, children }) => (
  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
      {title}
    </h3>
    {children}
  </div>
));

/** Stat card for the top summary row */
const StatCard = React.memo(({ label, value, icon: Icon, iconBg, iconColor }) => (
  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm text-gray-600 dark:text-gray-400">{label}</p>
        <p className="text-2xl font-bold text-gray-900 dark:text-white">{value}</p>
      </div>
      <div className={`${iconBg} rounded-lg p-3`}>
        <Icon className={`h-6 w-6 ${iconColor}`} />
      </div>
    </div>
  </div>
));

const Dashboard = () => {
  const { user } = useAuth();
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [period, setPeriod] = useState('week');


  const fetchEnergyStats = useCallback(async () => {
    try {
      const response = await apiFetch(`/api/energy/stats?period=${period}`);
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Failed to fetch energy stats:', error);
    } finally {
      setLoading(false);
    }
  }, [period]);

  useEffect(() => {
    fetchEnergyStats();
  }, [fetchEnergyStats]);

  // Memoize pie chart label renderer to avoid creating a new function every render
  const renderPieLabel = useCallback(
    ({ _id, percent }) => `${_id} ${(percent * 100).toFixed(0)}%`,
    []
  );

  // Memoize device consumption safely
  const deviceConsumption = useMemo(
    () => stats?.deviceConsumption ?? [],
    [stats?.deviceConsumption]
  );

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Welcome back, {user?.name}!
          </h1>
          <p className="text-gray-600 dark:text-gray-300 mt-2">
            Here's your energy consumption overview
          </p>
        </div>

        {/* Period Selector */}
        <div className="mb-8">
          <div className="flex space-x-2">
            {['week', 'month'].map((p) => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors duration-200 ${period === p
                  ? 'bg-blue-600 text-white'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                  }`}
              >
                {p.charAt(0).toUpperCase() + p.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <StatCard
            label="Total Energy"
            value={`${stats?.totalConsumption.total.toFixed(2)} kWh`}
            icon={Zap}
            iconBg="bg-blue-100 dark:bg-blue-900"
            iconColor="text-blue-600 dark:text-blue-400"
          />
          <StatCard
            label="Total Cost"
            value={`$${stats?.totalConsumption.totalCost.toFixed(2)}`}
            icon={TrendingUp}
            iconBg="bg-green-100 dark:bg-green-900"
            iconColor="text-green-600 dark:text-green-400"
          />
          <StatCard
            label="Avg. Daily"
            value={`${((stats?.totalConsumption.total || 0) / 7).toFixed(2)} kWh`}
            icon={TrendingDown}
            iconBg="bg-purple-100 dark:bg-purple-900"
            iconColor="text-purple-600 dark:text-purple-400"
          />
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Bar Chart */}
          <ChartCard title="Device Consumption">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={deviceConsumption}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="_id" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="totalEnergy" fill="#3B82F6" />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* Pie Chart */}
          <ChartCard title="Energy Distribution">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={deviceConsumption}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={renderPieLabel}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="totalEnergy"
                >
                  {deviceConsumption.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* Line Chart */}
        <ChartCard title="Daily Consumption Trend">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={stats?.dailyConsumption}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="_id" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="totalEnergy" stroke="#3B82F6" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Recommendations */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mt-8">
          <div className="flex items-center space-x-2 mb-4">
            <Lightbulb className="h-5 w-5 text-yellow-500" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Smart Recommendations
            </h3>
          </div>
          <div className="space-y-3">
            {stats?.recommendations.map((recommendation, index) => (
              <div
                key={index}
                className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4"
              >
                <p className="text-yellow-800 dark:text-yellow-300">{recommendation}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
