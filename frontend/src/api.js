import axios from 'axios';

// Single API URL - all endpoints are on the v2 backend
// For local dev: defaults to http://localhost:8001
// For VPS: set VITE_API_URL in .env to your VPS URL (e.g., https://yourdomain.com)
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';
const API_KEY = import.meta.env.VITE_API_KEY;

if (!API_KEY) {
  console.error('❌ VITE_API_KEY is not set! API requests will fail.');
}

const smartApi = axios.create({
  baseURL: API_URL,
  headers: {
    'X-API-Key': API_KEY,
  },
});

export const getStats = async () => {
  const response = await smartApi.get('/api/stats');
  return response.data;
};

export const search = async (filters, page = 1, pageSize = 50) => {
  const params = {
    page,
    page_size: pageSize,
    ...filters,
  };
  const response = await smartApi.get('/api/search', { params });
  return response.data;
};

export const getCountries = async () => {
  const response = await smartApi.get('/api/filters/countries');
  // Extract just the value strings for dropdown options
  return (response.data || []).map(item => typeof item === 'string' ? item : item.value);
};

export const getCities = async (country = null) => {
  const params = country ? { country } : {};
  const response = await smartApi.get('/api/filters/cities', { params });
  return (response.data || []).map(item => typeof item === 'string' ? item : item.value);
};

export const getIndustries = async () => {
  const response = await smartApi.get('/api/filters/industries');
  return (response.data || []).map(item => typeof item === 'string' ? item : item.value);
};

export const getLocations = async () => {
  const response = await smartApi.get('/api/filters/locations');
  // Locations need both location name and count for the dropdown display
  return (response.data || []).map(item => {
    if (typeof item === 'string') return { location: item, count: 0 };
    return { location: item.value || item.label, count: item.count || 0 };
  });
};

export const getSkills = async () => {
  const response = await smartApi.get('/api/filters/skills');
  return (response.data || []).map(item => typeof item === 'string' ? item : item.value);
};

export const getCertifications = async () => {
  const response = await smartApi.get('/api/filters/certifications');
  return (response.data || []).map(item => typeof item === 'string' ? item : item.value);
};

export const getSchools = async () => {
  const response = await smartApi.get('/api/filters/schools');
  return (response.data || []).map(item => typeof item === 'string' ? item : item.value);
};

export const getRoles = async () => {
  const response = await smartApi.get('/api/filters/roles');
  return (response.data || []).map(item => typeof item === 'string' ? item : item.value);
};

export const getPersonCertifications = async (forager_id) => {
  const response = await smartApi.get(`/api/person/${forager_id}/certifications`);
  return response.data;
};

export const getPersonEducations = async (forager_id) => {
  const response = await smartApi.get(`/api/person/${forager_id}/educations`);
  return response.data;
};

export const getPersonRoles = async (forager_id) => {
  const response = await smartApi.get(`/api/person/${forager_id}/roles`);
  return response.data;
};

export const getPersonSkills = async (forager_id) => {
  const response = await smartApi.get(`/api/person/${forager_id}/skills`);
  return response.data;
};

export const exportExcel = async (filters, limit = 100000) => {
  try {
    console.log('Starting export with filters:', filters);
    const response = await smartApi.get('/api/export/excel', {
      params: { ...filters, limit },
      responseType: 'blob',
      timeout: 120000, // 2 minute timeout for large exports
    });

    console.log('Export response received, size:', response.data.size);

    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `talentin_export_${Date.now()}.xlsx`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);

    console.log('Export completed successfully');
  } catch (error) {
    console.error('Export error details:', error);

    // Try to extract error message from response
    if (error.response && error.response.data instanceof Blob) {
      const text = await error.response.data.text();
      console.error('Error response:', text);
      try {
        const errorData = JSON.parse(text);
        throw new Error(errorData.detail || 'Export failed');
      } catch (e) {
        throw new Error(text || 'Export failed. Please check the backend logs.');
      }
    }

    throw new Error(error.response?.data?.detail || error.message || 'Export failed. Please try again with fewer results.');
  }
};

export const smartSearch = async (query, locationPreference = 'preferred', selectedLocations = []) => {
  try {
    const requestTimestamp = Date.now();
    const requestId = `${requestTimestamp}-${Math.random().toString(36).substr(2, 9)}`;
    
    console.log('='.repeat(60));
    console.log('🔍 FRONTEND: Making Smart Search Request');
    console.log('   Request ID:', requestId);
    console.log('   Query:', query);
    console.log('   Query Length:', query.length);
    console.log('   Location Preference:', locationPreference);
    console.log('   Selected Locations:', selectedLocations);
    console.log('   Timestamp:', new Date(requestTimestamp).toISOString());
    console.log('='.repeat(60));
    
    const response = await smartApi.post('/api/v2/smart-search', {
      query: query,
      limit: 50,
      location_preference: locationPreference,
      selected_locations: selectedLocations
    }, {
      headers: {
        'X-Request-ID': requestId  // Add unique request ID
      }
    });
    
    console.log('✅ FRONTEND: Smart Search Response Received');
    console.log('   Request ID:', requestId);
    console.log('   Response Time:', Date.now() - requestTimestamp, 'ms');
    console.log('   Total Matches:', response.data.total_matches);
    console.log('   Results Count:', response.data.results?.length || 0);
    if (response.data.results?.length > 0) {
      console.log('   First 3 Results:', response.data.results.slice(0, 3).map(r => r.full_name).join(', '));
    }
    console.log('='.repeat(60));

    // Adapt V2 response to match legacy format expected by CandidateCard
    const adaptedResults = response.data.results.map(r => {
      // Split full_name into first/last for display compatibility
      const nameParts = (r.full_name || 'Unknown Candidate').split(' ');
      const firstName = nameParts[0];
      const lastName = nameParts.slice(1).join(' ') || '';

      return {
        ...r,
        first_name: firstName,
        last_name: lastName,
        // Ensure location fields are populated
        city: r.city || r.location?.split(',')[0] || '',
        country: r.country || '',
        // Ensure photo is passed through
        photo: r.photo || null
      };
    });

    return {
      data: adaptedResults,
      total: response.data.total_matches,
      total_pages: Math.ceil(response.data.total_matches / 50) || 1,
      city_breakdown: response.data.city_breakdown || [],
      facets: response.data.facets || null,
      metadata: {
        parsing_method: response.data.query_understanding,
        took_ms: response.data.took_ms
      }
    };
  } catch (error) {
    console.error('Smart search failed:', error);
    throw error;
  }
};
