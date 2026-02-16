export const appliances = [
  {
    id: '1',
    name: 'Air Conditioner',
    category: 'HVAC',
    currentUsage: 2400,
    dailyUsage: 15.2,
    weeklyUsage: 106.4,
    monthlyUsage: 456,
    status: 'on',
    efficiency: 'medium',
    location: 'Living Room'
  },
  {
    id: '2',
    name: 'Refrigerator',
    category: 'Kitchen',
    currentUsage: 120,
    dailyUsage: 2.9,
    weeklyUsage: 20.3,
    monthlyUsage: 87,
    status: 'on',
    efficiency: 'high',
    location: 'Kitchen'
  },
  {
    id: '3',
    name: 'Washing Machine',
    category: 'Laundry',
    currentUsage: 0,
    dailyUsage: 1.2,
    weeklyUsage: 8.4,
    monthlyUsage: 36,
    status: 'off',
    efficiency: 'high',
    location: 'Laundry Room'
  },
  {
    id: '4',
    name: 'Water Heater',
    category: 'Utilities',
    currentUsage: 3500,
    dailyUsage: 8.4,
    weeklyUsage: 58.8,
    monthlyUsage: 252,
    status: 'oof',
    efficiency: 'low',
    location: 'Basement'
  },
  {
    id: '5',
    name: 'LED TV',
    category: 'Entertainment',
    currentUsage: 150,
    dailyUsage: 0.9,
    weeklyUsage: 6.3,
    monthlyUsage: 27,
    status: 'on',
    efficiency: 'high',
    location: 'Living Room'
  },
  {
    id: '6',
    name: 'Desktop Computer',
    category: 'Electronics',
    currentUsage: 250,
    dailyUsage: 2.0,
    weeklyUsage: 14.0,
    monthlyUsage: 60,
    status: 'on',
    efficiency: 'medium',
    location: 'Office'
  },
  {
    id: '7',
    name: 'Dishwasher',
    category: 'Kitchen',
    currentUsage: 0,
    dailyUsage: 0.8,
    weeklyUsage: 5.6,
    monthlyUsage: 24,
    status: 'standby',
    efficiency: 'high',
    location: 'Kitchen'
  },
  {
    id: '8',
    name: 'Electric Dryer',
    category: 'Laundry',
    currentUsage: 0,
    dailyUsage: 2.1,
    weeklyUsage: 14.7,
    monthlyUsage: 63,
    status: 'off',
    efficiency: 'medium',
    location: 'Laundry Room'
  }
];

export const hourlyData = [
  { hour: '00:00', usage: 1.2 },
  { hour: '01:00', usage: 1.1 },
  { hour: '02:00', usage: 1.0 },
  { hour: '03:00', usage: 0.9 },
  { hour: '04:00', usage: 0.8 },
  { hour: '05:00', usage: 0.9 },
  { hour: '06:00', usage: 1.5 },
  { hour: '07:00', usage: 2.8 },
  { hour: '08:00', usage: 3.2 },
  { hour: '09:00', usage: 2.9 },
  { hour: '10:00', usage: 2.5 },
  { hour: '11:00', usage: 2.7 },
  { hour: '12:00', usage: 3.1 },
  { hour: '13:00', usage: 3.4 },
  { hour: '14:00', usage: 3.8 },
  { hour: '15:00', usage: 4.2 },
  { hour: '16:00', usage: 4.5 },
  { hour: '17:00', usage: 5.1 },
  { hour: '18:00', usage: 5.8 },
  { hour: '19:00', usage: 6.2 },
  { hour: '20:00', usage: 5.9 },
  { hour: '21:00', usage: 5.3 },
  { hour: '22:00', usage: 4.1 },
  { hour: '23:00', usage: 2.8 }
];

export const dailyData = [
  { hour: 'Mon', usage: 45.2 },
  { hour: 'Tue', usage: 52.8 },
  { hour: 'Wed', usage: 48.1 },
  { hour: 'Thu', usage: 51.3 },
  { hour: 'Fri', usage: 49.7 },
  { hour: 'Sat', usage: 38.9 },
  { hour: 'Sun', usage: 42.3 }
];

export const usageStats = {
  today: 33.5,
  week: 328.3,
  month: 1342,
  cost: {
    today: 4.02,
    week: 39.40,
    month: 161.04
  }
};

export const recommendations = [
  {
    id: '1',
    appliance: 'Water Heater',
    type: 'timing',
    message: 'Run during 11 PM - 6 AM for 40% lower rates',
    savings: '$23/month',
    priority: 'high'
  },
  {
    id: '2',
    appliance: 'Air Conditioner',
    type: 'settings',
    message: 'Increase temperature by 2°F to save 15% energy',
    savings: '$18/month',
    priority: 'medium'
  },
  {
    id: '3',
    appliance: 'Washing Machine',
    type: 'timing',
    message: 'Use during off-peak hours (10 PM - 8 AM)',
    savings: '$8/month',
    priority: 'low'
  },
  {
    id: '4',
    appliance: 'Desktop Computer',
    type: 'usage',
    message: 'Enable sleep mode when idle for 15+ minutes',
    savings: '$12/month',
    priority: 'medium'
  },
  {
    id: '5',
    appliance: 'LED TV',
    type: 'standby',
    message: 'Unplug when not in use to eliminate phantom load',
    savings: '$4/month',
    priority: 'low'
  }
];
