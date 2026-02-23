import React from 'react';
import { IconDashboard, IconUsers } from './Icons';
import logo from '../assets/logo.png';

const ChevronLeft = ({ size = 20 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="15 18 9 12 15 6" />
    </svg>
);

const ChevronRight = ({ size = 20 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="9 18 15 12 9 6" />
    </svg>
);

const Sidebar = ({ activePage, onNavigate, collapsed, onToggle }) => {
    const menuItems = [
        { id: 'dashboard', label: 'Dashboard', icon: IconDashboard },
        { id: 'search', label: 'Leads', icon: IconUsers },
    ];

    return (
        <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
            <div className="sidebar-logo">
                {!collapsed && <img src={logo} alt="Talentin" className="logo-image" />}
                {collapsed && <span className="logo-icon">T</span>}
            </div>

            <nav className="sidebar-nav">
                {menuItems.map((item) => {
                    const Icon = item.icon;
                    const isActive = activePage === item.id;
                    return (
                        <button
                            key={item.id}
                            className={`nav-item ${isActive ? 'active' : ''}`}
                            onClick={() => onNavigate(item.id)}
                            title={collapsed ? item.label : ''}
                        >
                            <Icon size={20} />
                            {!collapsed && <span>{item.label}</span>}
                        </button>
                    );
                })}
            </nav>

            <div className="sidebar-footer">
                {!collapsed ? (
                    <div className="user-profile-mini">
                        <div className="avatar-mini">DA</div>
                        <div className="user-info-mini">
                            <div className="user-name">Demo Admin</div>
                            <div className="user-role">Admin</div>
                        </div>
                    </div>
                ) : (
                    <div className="user-profile-mini" style={{ justifyContent: 'center' }}>
                        <div className="avatar-mini" style={{ width: 32, height: 32, fontSize: '0.75rem' }}>DA</div>
                    </div>
                )}
            </div>

            <button
                className="sidebar-toggle"
                onClick={onToggle}
                title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            >
                {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
            </button>
        </aside>
    );
};

export default Sidebar;
