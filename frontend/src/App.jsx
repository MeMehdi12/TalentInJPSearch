import { useState, useEffect } from 'react';
import './App.css';
import { getStats, search, smartSearch, getCountries, getCities, getIndustries, getLocations, getSkills, getCertifications, getSchools, getRoles, exportExcel } from './api';
import Layout from './components/Layout';
import Header from './components/Header';
import LoginPage from './components/LoginPage';
import { getLoggedInUser, logout } from './auth';
import StatsCards from './components/StatsCards';
import CandidateCard from './components/CandidateCard';
import SmartSearchBar from './components/SmartSearchBar';
import { IconSearch, IconFilter, IconX, IconDownload, IconLoader, IconMapPin, IconGlobe, IconTarget, IconChevronDown } from './components/Icons';
import { DashboardCharts } from './components/Charts';

function App() {
  const [user, setUser] = useState(() => getLoggedInUser());
  const [activePage, setActivePage] = useState('dashboard');
  const [stats, setStats] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [exporting, setExporting] = useState(false);
  const [searchMode, setSearchMode] = useState('classic'); // 'classic' | 'smart'

  // City filtering state
  const [allResults, setAllResults] = useState([]); // Store all results before filtering
  const [cityBreakdown, setCityBreakdown] = useState([]); // {city: "SF", count: 25}
  const [selectedCities, setSelectedCities] = useState(new Set());
  const [locationPreference, setLocationPreference] = useState('preferred'); // 'remote', 'preferred', 'must_match'
  const [showLocDropdown, setShowLocDropdown] = useState(false);
  const [citySearch, setCitySearch] = useState('');

  // Filters state
  const [filters, setFilters] = useState({
    first_name: '',
    last_name: '',
    country: '',
    city: '',
    industry: '',
    area: '',
    location: '',
    headline: '',
    skill: '',
    certification: '',
    education: '',
    role: '',
  });
  const [quickSearch, setQuickSearch] = useState('');

  const [countries, setCountries] = useState([]);
  const [cities, setCities] = useState([]);
  const [industries, setIndustries] = useState([]);
  const [locations, setLocations] = useState([]);
  const [skills, setSkills] = useState([]);
  const [certifications, setCertifications] = useState([]);
  const [schools, setSchools] = useState([]);
  const [roles, setRoles] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(50);
  const [showFilters, setShowFilters] = useState(false);

  useEffect(() => {
    loadStats();
    loadDropdownOptions();
  }, []);

  const loadStats = async () => {
    try {
      const data = await getStats();
      setStats(data);
    } catch (err) {
      console.error('Failed to load stats:', err);
    }
  };

  const loadDropdownOptions = async () => {
    try {
      const [countriesData, industriesData, locationsData, skillsData, certificationsData, schoolsData, rolesData] = await Promise.all([
        getCountries(),
        getIndustries(),
        getLocations(),
        getSkills(),
        getCertifications(),
        getSchools(),
        getRoles()
      ]);
      setCountries(countriesData);
      setIndustries(industriesData);
      setLocations(locationsData);
      setSkills(skillsData);
      setCertifications(certificationsData);
      setSchools(schoolsData);
      setRoles(rolesData);
    } catch (err) {
      console.error('Failed to load dropdown options:', err);
    }
  };

  useEffect(() => {
    if (filters.country) {
      getCities(filters.country).then(setCities).catch(console.error);
    } else {
      setCities([]);
    }
  }, [filters.country]);

  const handleFilterChange = (field, value) => {
    setFilters(prev => ({ ...prev, [field]: value }));
  };

  const handleSearch = async () => {
    // CRITICAL: Clear previous results FIRST to prevent mixing
    setResults(null);
    setAllResults([]);
    setCityBreakdown([]);
    setFacets(null);
    
    setLoading(true);
    setError(null);
    setCurrentPage(1);

    console.log('🔍 FRONTEND: Starting Classic Search', { filters, quickSearch });

    try {
      const activeFilters = Object.fromEntries(
        Object.entries(filters).filter(([_, v]) => v !== '')
      );
      // Include quick search text as a separate parameter
      if (quickSearch.trim()) {
        activeFilters.quick_search = quickSearch.trim();
      }
      const data = await search(activeFilters, 1, pageSize);
      console.log('✅ FRONTEND: Classic Search Complete', { 
        total: data.total,
        returned: data.data?.length,
        firstResult: data.data?.[0]?.full_name 
      });
      
      setResults(data);
      if (activePage !== 'search') setActivePage('search');
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to search. Check your API key and connection.');
      console.error('❌ Classic search error:', err);
    } finally {
      setLoading(false);
    }
  };

  const [currentQuery, setCurrentQuery] = useState('');
  const [facets, setFacets] = useState(null);

  const handleSmartSearch = async (query, overridePreference = null) => {
    // CRITICAL: Clear previous results FIRST to prevent mixing
    setAllResults([]);
    setCityBreakdown([]);
    setFacets(null);
    setResults(null);
    
    setLoading(true);
    setError(null);
    setCurrentPage(1);
    setSelectedCities(new Set());

    // Save query for re-running
    setCurrentQuery(query);
    const pref = overridePreference || locationPreference;

    console.log('🔍 FRONTEND: Starting Smart Search', { query, pref });

    try {
      const data = await smartSearch(query, pref);
      console.log('✅ FRONTEND: Smart Search Complete', { 
        total: data.total, 
        returned: data.data?.length,
        firstResult: data.data?.[0]?.full_name 
      });
      
      setAllResults(data.data || []);
      setCityBreakdown(data.city_breakdown || []);
      setFacets(data.facets || null);
      setResults(data);
      if (activePage !== 'search') setActivePage('search');
    } catch (err) {
      setError('Smart search failed. Please try again.');
      console.error('❌ Smart search error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handlePageChange = async (newPage) => {
    if (newPage < 1 || (results && newPage > results.total_pages)) return;
    setLoading(true);
    setCurrentPage(newPage);
    try {
      const activeFilters = Object.fromEntries(
        Object.entries(filters).filter(([_, v]) => v !== '')
      );
      if (quickSearch.trim()) {
        activeFilters.quick_search = quickSearch.trim();
      }
      const data = await search(activeFilters, newPage, pageSize);
      setResults(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load page.');
      console.error('Page change error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleClearFilters = () => {
    setFilters({
      first_name: '',
      last_name: '',
      country: '',
      city: '',
      industry: '',
      area: '',
      location: '',
      headline: '',
      skill: '',
      certification: '',
      education: '',
      role: '',
    });
    setQuickSearch('');
    setResults(null);
    setCurrentPage(1);
    setSelectedCities(new Set());
    setIsRemote(false);
    setCityBreakdown([]);
    setAllResults([]);
  };

  const toggleCityFilter = (city) => {
    if (locationPreference === 'remote') return; // Cannot select cities if Remote is on
    const newSelected = new Set(selectedCities);
    if (newSelected.has(city)) {
      newSelected.delete(city);
    } else {
      newSelected.add(city);
    }
    setSelectedCities(newSelected);
  };

  const removeFilter = (type, value) => {
    if (type === 'location_preference') setLocationPreference('preferred');
    if (type === 'city') toggleCityFilter(value);
  };

  // Get filtered results based on selected cities
  const getFilteredResults = () => {
    // Remote mode: show ALL candidates (no location filtering)
    if (locationPreference === 'remote') {
      return allResults;
    }
    // No cities selected: show all
    if (selectedCities.size === 0) {
      return allResults;
    }
    // Cities selected: filter by selected cities
    return allResults.filter(r => selectedCities.has(r.city));
  };

  const handleExportExcel = async () => {
    // ... (same)
    if (!results || results.total === 0) {
      alert('No results to export. Please search first.');
      return;
    }

    const confirmExport = results.total > 50000
      ? confirm(`You're about to export ${results.total.toLocaleString()} records. This may take a few minutes. Continue?`)
      : true;

    if (!confirmExport) return;

    setExporting(true);
    try {
      const activeFilters = Object.fromEntries(
        Object.entries(filters).filter(([_, v]) => v !== '')
      );
      await exportExcel(activeFilters);
      alert(`Successfully exported! Check your downloads folder.`);
    } catch (err) {
      alert('Export failed: ' + (err.message || 'Please try again or export fewer results.'));
      console.error('Export error:', err);
    } finally {
      setExporting(false);
    }
  };

  const switchSearchMode = (mode) => {
    if (mode === searchMode) return; // Already in this mode
    
    console.log(`🔄 SWITCHING SEARCH MODE: ${searchMode} → ${mode}`);
    
    // CRITICAL: Clear ALL state to prevent result mixing
    setSearchMode(mode);
    setResults(null);
    setCurrentPage(1);
    setError(null);
    setLoading(false);
    setQuickSearch('');
    setCurrentQuery('');
    setAllResults([]);
    setCityBreakdown([]);
    setSelectedCities(new Set());
    setFacets(null);
    setShowFilters(false);
    setCitySearch('');
    setFilters({
      first_name: '',
      last_name: '',
      country: '',
      city: '',
      industry: '',
      area: '',
      location: '',
      headline: '',
      skill: '',
      certification: '',
      education: '',
      role: '',
    });
    
    console.log('✅ Search mode switched, all state cleared');
  };

  const renderSearchFilters = () => (
    <div className={`filter-bar ${showFilters ? 'expanded' : ''} ${searchMode === 'smart' ? 'with-smart-search' : ''}`}>
      <div className="filter-row">
        {/* Search Mode Toggle */}
        <div className="mode-toggle" style={{ marginRight: '1rem', display: 'flex' }}>
          <button
            className={`btn-filter-toggle ${searchMode === 'classic' ? 'active' : ''}`}
            onClick={() => switchSearchMode('classic')}
            style={{ borderRadius: '6px 0 0 6px', borderRight: '1px solid rgba(255,255,255,0.1)' }}
          >
            Classic
          </button>
          <button
            className={`btn-filter-toggle ${searchMode === 'smart' ? 'active' : ''}`}
            onClick={() => switchSearchMode('smart')}
            style={{ borderRadius: '0 6px 6px 0' }}
          >
            ✨ Smart
          </button>
        </div>

        {searchMode === 'classic' ? (
          <>
            <div className="search-input-group">
              <IconSearch className="search-icon" size={20} />
              <input
                type="text"
                placeholder="Quick search: name, job title, company..."
                value={quickSearch}
                onChange={(e) => setQuickSearch(e.target.value)}
                className="main-search-input"
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              />
            </div>

            <button
              className={`btn-filter-toggle ${showFilters ? 'active' : ''}`}
              onClick={() => setShowFilters(!showFilters)}
            >
              <IconFilter size={18} /> Filters
            </button>

            <button className="btn-primary" onClick={handleSearch} disabled={loading}>
              {loading ? 'Searching...' : 'Search'}
            </button>
          </>
        ) : (
          <SmartSearchBar
            onSearch={handleSmartSearch}
            loading={loading}
          />
        )}
      </div>

      {/* Location Preference Dropdown (Below search bar) */}
      {searchMode === 'smart' && (
        <div className="search-sub-row">
          <div className="custom-dropdown-container">
            <div
              className="dropdown-trigger"
              onClick={() => setShowLocDropdown(!showLocDropdown)}
            >
              <div className="dropdown-trigger-content">
                <span className="dropdown-icon-wrapper">
                  {locationPreference === 'remote' && <IconGlobe size={18} />}
                  {locationPreference === 'preferred' && <IconMapPin size={18} />}
                  {locationPreference === 'must_match' && <IconTarget size={18} />}
                </span>
                <span>
                  {locationPreference === 'remote' && "Remote Position"}
                  {locationPreference === 'preferred' && "Location Match Preferred"}
                  {locationPreference === 'must_match' && "Location Must Match"}
                </span>
              </div>
              <IconChevronDown size={16} />
            </div>

            {showLocDropdown && (
              <div className="dropdown-menu">
                <div
                  className={`dropdown-item ${locationPreference === 'remote' ? 'selected' : ''}`}
                  onClick={() => {
                    setLocationPreference('remote');
                    setShowLocDropdown(false);
                    if (currentQuery) handleSmartSearch(currentQuery, 'remote');
                  }}
                >
                  <span className="dropdown-icon-wrapper"><IconGlobe size={18} /></span>
                  <span>Remote Position</span>
                </div>
                <div
                  className={`dropdown-item ${locationPreference === 'preferred' ? 'selected' : ''}`}
                  onClick={() => {
                    setLocationPreference('preferred');
                    setShowLocDropdown(false);
                    if (currentQuery) handleSmartSearch(currentQuery, 'preferred');
                  }}
                >
                  <span className="dropdown-icon-wrapper"><IconMapPin size={18} /></span>
                  <span>Location Match Preferred</span>
                </div>
                <div
                  className={`dropdown-item ${locationPreference === 'must_match' ? 'selected' : ''}`}
                  onClick={() => {
                    setLocationPreference('must_match');
                    setShowLocDropdown(false);
                    if (currentQuery) handleSmartSearch(currentQuery, 'must_match');
                  }}
                >
                  <span className="dropdown-icon-wrapper"><IconTarget size={18} /></span>
                  <span>Location Must Match</span>
                </div>
              </div>
            )}
            {/* Overlay to close dropdown when clicking outside */}
            {showLocDropdown && (
              <div
                style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, zIndex: 90 }}
                onClick={() => setShowLocDropdown(false)}
              />
            )}
          </div>
        </div>
      )}

      {/* Only show filters in classic mode */}
      {
        searchMode === 'classic' && showFilters && (
          <div className="filter-details">
            <div className="filter-grid">
              <div className="form-group">
                <label>First Name</label>
                <input
                  type="text"
                  value={filters.first_name}
                  onChange={(e) => handleFilterChange('first_name', e.target.value)}
                  placeholder="First Name"
                />
              </div>
              <div className="form-group">
                <label>Last Name</label>
                <input
                  type="text"
                  value={filters.last_name}
                  onChange={(e) => handleFilterChange('last_name', e.target.value)}
                  placeholder="Last Name"
                />
              </div>
              <div className="form-group">
                <label>Country</label>
                <select
                  value={filters.country}
                  onChange={(e) => handleFilterChange('country', e.target.value)}
                >
                  <option value="">All Countries</option>
                  {countries.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>City</label>
                <select
                  value={filters.city}
                  onChange={(e) => handleFilterChange('city', e.target.value)}
                  disabled={!filters.country}
                >
                  <option value="">All Cities</option>
                  {cities.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>Industry</label>
                <select
                  value={filters.industry}
                  onChange={(e) => handleFilterChange('industry', e.target.value)}
                >
                  <option value="">All Industries</option>
                  {industries.map(i => <option key={i} value={i}>{i}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>Area</label>
                <input
                  type="text"
                  value={filters.area}
                  onChange={(e) => handleFilterChange('area', e.target.value)}
                  placeholder="Area"
                />
              </div>
              <div className="form-group">
                <label>Headline</label>
                <input
                  type="text"
                  value={filters.headline}
                  onChange={(e) => handleFilterChange('headline', e.target.value)}
                  placeholder="Job title, keywords..."
                />
              </div>
              <div className="form-group">
                <label>Location (Any)</label>
                <select
                  value={filters.location}
                  onChange={(e) => handleFilterChange('location', e.target.value)}
                >
                  <option value="">All Locations</option>
                  {locations.map((loc, idx) => (
                    <option key={idx} value={loc.location}>
                      {loc.location} ({loc.count.toLocaleString()})
                    </option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label>Skill</label>
                <select
                  value={filters.skill}
                  onChange={(e) => handleFilterChange('skill', e.target.value)}
                >
                  <option value="">Any Skill</option>
                  {skills.map((s, idx) => <option key={idx} value={s}>{s}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>Certification</label>
                <select
                  value={filters.certification}
                  onChange={(e) => handleFilterChange('certification', e.target.value)}
                >
                  <option value="">Any Certification</option>
                  {certifications.map((c, idx) => <option key={idx} value={c}>{c}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>School / Education</label>
                <select
                  value={filters.education}
                  onChange={(e) => handleFilterChange('education', e.target.value)}
                >
                  <option value="">Any School</option>
                  {schools.map((s, idx) => <option key={idx} value={s}>{s}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>Job Title (Past Positions)</label>
                <select
                  value={filters.role}
                  onChange={(e) => handleFilterChange('role', e.target.value)}
                >
                  <option value="">Any Job Title</option>
                  {roles.map((r, idx) => <option key={idx} value={r}>{r}</option>)}
                </select>
              </div>
            </div>
            <div className="filter-actions">
              <button className="btn-text" onClick={handleClearFilters}>Clear all</button>
            </div>
          </div>
        )
      }
    </div >
  );

  const handleLogout = () => {
    logout();
    setUser(null);
  };

  if (!user) {
    return <LoginPage onLogin={(email) => setUser(email)} />;
  }

  return (
    <Layout activePage={activePage} onNavigate={setActivePage} user={user}>
      <div className="page-content">
        {activePage === 'dashboard' && (
          <>
            <Header title="Dashboard" subtitle="Welcome back, Demo Admin" user={user} onLogout={handleLogout} />
            <StatsCards stats={stats} />
            <DashboardCharts />
          </>
        )}

        {activePage === 'search' && (
          <>
            <Header title="Leads" subtitle="Find and qualify potential customers" user={user} onLogout={handleLogout} />
            {renderSearchFilters()}

            <div className="results-area">
              {error && <div className="error-message">{error}</div>}

              {results && (
                <div className={`results-container ${searchMode === 'smart' && cityBreakdown.length > 0 ? 'with-sidebar' : ''}`}>
                  {/* SIDEBAR FILTER */}
                  {searchMode === 'smart' && cityBreakdown.length > 0 && (
                    <div className="city-filter-sidebar">
                      <div className="sidebar-main-header">
                        <h3>Filters</h3>
                        {(selectedCities.size > 0 || locationPreference !== 'preferred') && (
                          <button className="btn-clear-all-text" onClick={() => { setLocationPreference('preferred'); setSelectedCities(new Set()); }}>
                            Clear all
                          </button>
                        )}
                      </div>

                      <div className="sidebar-section">
                        <h4>Location</h4>

                        {locationPreference === 'remote' ? (
                          <div className="filter-disabled-msg">
                            <IconMapPin size={16} />
                            <span>Location filters disabled for remote search</span>
                          </div>
                        ) : locationPreference === 'must_match' ? (
                          <div className="filter-disabled-msg">
                            <IconMapPin size={16} />
                            <span style={{ color: '#ff6b35' }}>Showing ONLY candidates from target location</span>
                          </div>
                        ) : (
                          <>
                            {cityBreakdown.length > 8 && (
                              <input
                                type="text"
                                placeholder="Search cities..."
                                className="city-search-input"
                                value={citySearch}
                                onChange={(e) => setCitySearch(e.target.value)}
                              />
                            )}
                            <div className="filter-checklist">
                              {cityBreakdown
                                .filter(c => c.count > 0) // Hide 0 count
                                .sort((a, b) => b.count - a.count) // Sort by count desc
                                .filter(c => c.city.toLowerCase().includes(citySearch.toLowerCase()))
                                .map(({ city, count }) => (
                                  <label key={city} className="checklist-item">
                                    <input
                                      type="checkbox"
                                      checked={selectedCities.has(city)}
                                      onChange={() => toggleCityFilter(city)}
                                    />
                                    <span className="checklist-label">{city.replace(/\b\w/g, c => c.toUpperCase())}</span>
                                    <span className="checklist-count">{count}</span>
                                  </label>
                                ))}
                            </div>
                          </>
                        )}
                      </div>

                      {/* SKILLS FACET */}
                      {facets && facets.skills && facets.skills.length > 0 && (
                        <div className="sidebar-section">
                          <h4>Top Skills</h4>
                          <div className="filter-checklist">
                            {facets.skills.slice(0, 10).map((skill) => (
                              <div key={skill.value} className="checklist-item-readonly">
                                <span className="checklist-label">{skill.value}</span>
                                <span className="checklist-count">{skill.count}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* RESULTS CONTENT */}
                  <div className="results-content">

                    {/* Active Chips Bar */}
                    {(locationPreference !== 'preferred' || selectedCities.size > 0) && (
                      <div className="active-filters-bar">
                        {locationPreference === 'remote' && (
                          <div className="filter-chip" onClick={() => removeFilter('location_preference')}>
                            🌍 Remote Position <IconX size={14} />
                          </div>
                        )}
                        {locationPreference === 'must_match' && (
                          <div className="filter-chip" onClick={() => removeFilter('location_preference')}>
                            🎯 Location Must Match <IconX size={14} />
                          </div>
                        )}
                        {Array.from(selectedCities).map(city => (
                          <div key={city} className="filter-chip" onClick={() => removeFilter('city', city)}>
                            {city.replace(/\b\w/g, c => c.toUpperCase())} <IconX size={14} />
                          </div>
                        ))}
                      </div>
                    )}

                    <div className="results-header-bar">
                      <span className="results-count">
                        <strong>{(searchMode === 'smart' && (selectedCities.size > 0 || locationPreference !== 'preferred') ? getFilteredResults().length : results.total).toLocaleString()}</strong> leads found
                        {searchMode === 'smart' && (selectedCities.size > 0 || locationPreference !== 'preferred') && (
                          <span className="filter-note"> (filtered from {allResults.length})</span>
                        )}
                      </span>
                      <div className="results-actions">
                        <button
                          className="btn-export"
                          onClick={handleExportExcel}
                          disabled={exporting || loading}
                          title="Export up to 100,000 filtered results"
                        >
                          {exporting ? (
                            <>
                              <IconLoader size={18} className="spinning" />
                              Exporting...
                            </>
                          ) : (
                            <>
                              <IconDownload size={18} />
                              Export to Excel
                            </>
                          )}
                        </button>
                      </div>
                    </div>

                    <div className="candidates-grid">
                      {(searchMode === 'smart' ? getFilteredResults() : results.data)
                        .filter((person, index, self) =>
                          index === self.findIndex((p) => (
                            p.forager_id === person.forager_id ||
                            (p.first_name === person.first_name && p.last_name === person.last_name && p.headline === person.headline)
                          ))
                        )
                        .map((person, idx) => (
                          <CandidateCard 
                            key={`${searchMode}-${person.forager_id || person.full_name || idx}`} 
                            person={person} 
                          />
                        ))}
                    </div>

                    <div className="pagination">
                      <button
                        className="btn-pagination"
                        onClick={() => handlePageChange(currentPage - 1)}
                        disabled={currentPage === 1 || loading}
                      >
                        Previous
                      </button>
                      <span className="page-info">
                        Page {currentPage} of {results.total_pages}
                      </span>
                      <button
                        className="btn-pagination"
                        onClick={() => handlePageChange(currentPage + 1)}
                        disabled={currentPage === results.total_pages || loading}
                      >
                        Next
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </>
        )}

        {activePage === 'campaigns' && (
          <>
            <Header title="Campaigns" subtitle="Manage your outreach campaigns" user={user} onLogout={handleLogout} />
            <div className="empty-state">
              <h3>No active campaigns</h3>
              <p>Create a campaign to start reaching out to leads.</p>
            </div>
          </>
        )}
      </div>
    </Layout >
  );
}

export default App;
