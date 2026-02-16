import Device from '../models/Device.js';
import EnergyUsage from '../models/EnergyUsage.js';

export const createSampleData = async (userId) => {
  try {
    // Create sample devices
    const sampleDevices = [
      { name: 'Living Room AC', type: 'AC', wattage: 1500, location: 'Living Room' },
      { name: 'Samsung 55" TV', type: 'TV', wattage: 150, location: 'Living Room' },
      { name: 'Space Heater', type: 'Heater', wattage: 1200, location: 'Bedroom' },
      { name: 'Kitchen Fridge', type: 'Fridge', wattage: 300, location: 'Kitchen' },
      { name: 'Washing Machine', type: 'Washing Machine', wattage: 800, location: 'Utility Room' },
      { name: 'LED Lights', type: 'Lights', wattage: 60, location: 'Living Room' }
    ];

    const devices = await Device.insertMany(
      sampleDevices.map(device => ({ ...device, userId }))
    );

    // Create sample energy usage data for the past week
    const energyUsageData = [];
    const now = new Date();

    for (let i = 0; i < 7; i++) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      
      devices.forEach(device => {
        const hoursUsed = Math.random() * 8 + 2; // Random hours between 2-10
        const energyConsumed = (device.wattage * hoursUsed) / 1000; // kWh
        const cost = energyConsumed * 0.12; // $0.12 per kWh

        energyUsageData.push({
          userId,
          deviceId: device._id,
          deviceName: device.name,
          wattage: device.wattage,
          hoursUsed,
          energyConsumed,
          cost,
          date
        });
      });
    }

    await EnergyUsage.insertMany(energyUsageData);
    console.log('Sample data created successfully');
  } catch (error) {
    console.error('Error creating sample data:', error);
  }
};