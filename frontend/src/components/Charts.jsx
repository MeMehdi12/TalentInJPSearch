import React, { useEffect, useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL;
const API_KEY = import.meta.env.VITE_API_KEY;

// Bar Chart Component with Logarithmic Scale
export const BarChart = ({ data, title, color = '#00a884' }) => {
  if (!data || data.length === 0) return null;

  // Use logarithmic scale for height calculation to handle large outliers
  const maxVal = Math.max(...data.map(d => d.value));
  const minVal = Math.min(...data.map(d => d.value));
  const chartHeight = 200;

  // Helper to calculate height
  const getHeight = (val) => {
    if (val === 0) return 0;
    // Log normalization: (log(val) / log(max)) * height
    // We add 1 to avoid log(0) and ensure base baseline
    const logMax = Math.log(maxVal + 1);
    const logVal = Math.log(val + 1);
    const height = (logVal / logMax) * chartHeight;
    return Math.max(height, 4); // Minimum 4px height for visibility
  };

  return (
    <div className="chart-container">
      <h3 className="chart-title">{title}</h3>
      <div className="bar-chart">
        {data.slice(0, 10).map((item, index) => (
          <div key={index} className="bar-item" title={`${item.name}: ${item.value.toLocaleString()}`}>
            <div className="bar-wrapper">
              <div
                className="bar-fill"
                style={{
                  height: `${getHeight(item.value)}px`,
                  backgroundColor: color
                }}
              >
                <span className="bar-value short">{item.value >= 1000 ? (item.value / 1000).toFixed(1) + 'k' : item.value}</span>
              </div>
            </div>
            <div className="bar-label">{item.name}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Donut Chart Component
export const DonutChart = ({ data, title }) => {
  if (!data || data.length === 0) return null;

  const total = data.reduce((sum, item) => sum + item.value, 0);
  const colors = ['#00a884', '#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b', '#fa709a'];

  let currentAngle = 0;
  const segments = data.slice(0, 8).map((item, index) => {
    const percentage = (item.value / total) * 100;
    const angle = (percentage / 100) * 360;
    const startAngle = currentAngle;
    currentAngle += angle;

    // Calculate path for donut segment
    const radius = 80;
    const innerRadius = 50;

    // Fix for 100% single segment
    if (percentage >= 100) {
      return {
        path: `M 100 20 A 80 80 0 1 1 99.99 20 Z M 100 50 A 50 50 0 1 0 99.99 50 Z`,
        color: colors[0],
        name: item.name,
        value: item.value,
        percentage: '100.0'
      };
    }

    const startRad = (startAngle - 90) * Math.PI / 180;
    const endRad = (currentAngle - 90) * Math.PI / 180;

    const x1 = 100 + radius * Math.cos(startRad);
    const y1 = 100 + radius * Math.sin(startRad);
    const x2 = 100 + radius * Math.cos(endRad);
    const y2 = 100 + radius * Math.sin(endRad);
    const x3 = 100 + innerRadius * Math.cos(endRad);
    const y3 = 100 + innerRadius * Math.sin(endRad);
    const x4 = 100 + innerRadius * Math.cos(startRad);
    const y4 = 100 + innerRadius * Math.sin(startRad);

    const largeArc = angle > 180 ? 1 : 0;

    const pathData = [
      `M ${x1} ${y1}`,
      `A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`,
      `L ${x3} ${y3}`,
      `A ${innerRadius} ${innerRadius} 0 ${largeArc} 0 ${x4} ${y4}`,
      'Z'
    ].join(' ');

    return {
      path: pathData,
      color: colors[index % colors.length],
      name: item.name,
      value: item.value,
      percentage: percentage.toFixed(1)
    };
  });

  return (
    <div className="chart-container">
      <h3 className="chart-title">{title}</h3>
      <div className="donut-chart-wrapper">
        <div className="donut-visual">
          <svg viewBox="0 0 200 200" className="donut-svg">
            {segments.map((segment, index) => (
              <path
                key={index}
                d={segment.path}
                fill={segment.color}
                className="donut-segment"
              >
                <title>{`${segment.name}: ${segment.value.toLocaleString()} (${segment.percentage}%)`}</title>
              </path>
            ))}
            <text x="100" y="95" textAnchor="middle" className="donut-center-text" fontSize="24" fontWeight="bold">
              {total.toLocaleString()}
            </text>
            <text x="100" y="110" textAnchor="middle" className="donut-center-label" fontSize="12" fill="#999">
              Total
            </text>
          </svg>
        </div>
        <div className="donut-legend">
          {segments.map((segment, index) => (
            <div key={index} className="legend-item">
              <span className="legend-color" style={{ backgroundColor: segment.color }}></span>
              <span className="legend-label" title={segment.name}>{segment.name}</span>
              <span className="legend-value">{segment.percentage}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Dashboard Charts Component
export const DashboardCharts = () => {
  const [chartData, setChartData] = useState({
    countries: [],
    cities: [],
    industries: []
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadChartData();
  }, []);

  const loadChartData = async () => {
    try {
      if (!API_URL) {
        throw new Error("VITE_API_URL is missing in environment variables.");
      }

      const headers = {};
      if (API_KEY) {
        headers['X-API-Key'] = API_KEY;
      } else {
        console.warn("VITE_API_KEY is missing. Requests might fail if auth is required.");
      }

      const [countriesRes, industriesRes] = await Promise.all([
        axios.get(`${API_URL}/api/analytics/countries`, { headers }),
        axios.get(`${API_URL}/api/analytics/industries`, { headers })
      ]);

      const formatName = (str) => {
        if (!str) return '';
        return str.replace(/\b\w/g, l => l.toUpperCase());
      };

      const topCountries = countriesRes.data.map(item => ({
        name: formatName(item.country),
        value: item.count
      }));

      const topIndustries = industriesRes.data.map(item => ({
        name: formatName(item.industry),
        value: item.count
      }));

      setChartData({
        countries: topCountries,
        cities: [],
        industries: topIndustries
      });
      setError(null);
    } catch (error) {
      console.error('Failed to load chart data:', error);
      setError("Failed to load chart data. Please check connection.");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="charts-loading">
        <div className="loading-spinner"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-charts">
        <div className="error-message">
          {error}
        </div>
      </div>
    )
  }

  return (
    <div className="dashboard-charts">
      <div className="charts-grid">
        <BarChart
          data={chartData.countries}
          title="Top Countries"
          color="#00a884"
        />
        <DonutChart
          data={chartData.industries}
          title="Industry Distribution"
        />
      </div>
    </div>
  );
};
