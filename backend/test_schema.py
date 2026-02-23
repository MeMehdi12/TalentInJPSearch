import duckdb
conn = duckdb.connect("d:/Talentin JPSEARCH DEPLOY/data/talentin_profiles.duckdb", read_only=True)
print("certifications columns:", [c[0] for c in conn.execute("DESCRIBE certifications").fetchall()])
print("educations columns:", [c[0] for c in conn.execute("DESCRIBE educations").fetchall()])
print("roles columns:", [c[0] for c in conn.execute("DESCRIBE roles").fetchall()])
