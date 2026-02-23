import duckdb
conn = duckdb.connect('/var/www/talentin/data/talentin_profiles.duckdb', read_only=True)
print("certifications:")
print([c[0] for c in conn.execute('DESCRIBE certifications').fetchall()])
print("educations:")
print([c[0] for c in conn.execute('DESCRIBE educations').fetchall()])
print("roles:")
print([c[0] for c in conn.execute('DESCRIBE roles').fetchall()])
