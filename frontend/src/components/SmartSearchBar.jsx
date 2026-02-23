import React, { useState } from 'react';
import { IconSearch, IconLoader, IconX } from './Icons';
import './SmartSearchBar.css';

const SmartSearchBar = ({ onSearch, loading }) => {
    const [query, setQuery] = useState('');
    const [isFocused, setIsFocused] = useState(false);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (query.trim()) {
            onSearch(query);
        }
    };

    const handleClear = () => {
        setQuery('');
        // Optional: Trigger search reset if desired, or just clear input
    };

    return (
        <div className={`smart-search-container ${isFocused ? 'focused' : ''}`}>
            <div className="smart-search-wrapper">
                <div className="smart-ai-badge">
                    <span className="ai-sparkle"></span> AI-Powered
                </div>
                <form onSubmit={handleSubmit} className="smart-search-form">
                    <div className="input-wrapper">
                        <IconSearch className="search-icon-input" size={20} />
                        <input
                            type="text"
                            className="smart-search-input"
                            placeholder="Describe your ideal candidate..."
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            onFocus={() => setIsFocused(true)}
                            onBlur={() => setIsFocused(false)}
                            disabled={loading}
                        />
                        {query && !loading && (
                            <button type="button" className="btn-clear" onClick={handleClear}>
                                <IconX size={16} />
                            </button>
                        )}
                    </div>
                    <button type="submit" className="smart-search-btn" disabled={loading || !query.trim()}>
                        {loading ? (
                            <><IconLoader className="spinning" size={20} /> Analyzing...</>
                        ) : (
                            <>Find Top Candidates</>
                        )}
                    </button>
                </form>

                <div className="smart-search-hints">
                    <span className="hint-label">Try:</span>
                    <button type="button" className="hint-chip" onClick={() => setQuery("Senior Python Dev in SF")}>
                        Senior Python Dev in SF
                    </button>
                    <button type="button" className="hint-chip" onClick={() => setQuery("USCPA with 5+ years experience")}>
                        USCPA with 5+ years experience
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SmartSearchBar;
