import React from 'react';

const StatsCards = ({ stats }) => {
    if (!stats) return null;

    const cards = [
        { label: 'Total Leads', value: stats.total_records || 0, trend: 'up' },
        { label: 'Countries', value: stats.unique_countries || 0, trend: 'neutral' },
        { label: 'Cities', value: stats.unique_cities || 0, trend: 'up' },
        { label: 'Total Skills', value: stats.total_skills || 0, trend: 'up' },
    ];

    return (
        <div className="stats-grid">
            {cards.map((card, index) => (
                <div key={index} className="stat-card">
                    <div className="stat-content">
                        <div className="stat-label">{card.label}</div>
                        <div className="stat-value">{card.value.toLocaleString()}</div>
                    </div>
                    <div className="stat-chart">
                        {/* Simple SVG sparkline for visual effect */}
                        <svg width="100" height="40" viewBox="0 0 100 40">
                            <path
                                d="M0 30 Q 20 35, 40 20 T 100 10"
                                fill="none"
                                stroke={index % 2 === 0 ? '#00a884' : '#667eea'}
                                strokeWidth="2"
                            />
                        </svg>
                    </div>
                </div>
            ))}
        </div>
    );
};

export default StatsCards;
