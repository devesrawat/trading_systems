from composio import Composio

# Using the provided API key
api_key = "uak_6wyfKhEYHS1Z1NdsUg6z"
composio = Composio(api_key=api_key)

try:
    print("Testing Angel One connection...")
    # List connected accounts
    accounts = composio.connected_accounts.get()
    print(f"Connected accounts: {accounts}")

    # Try to find Angel One account
    angel_account = next((a for a in accounts if "angel" in a.app_slug.lower()), None)
    if angel_account:
        print(f"Found Angel One account: {angel_account.id} (Status: {angel_account.status})")
    else:
        print("Angel One account not found in connected accounts.")

except Exception as e:
    print(f"Error: {e}")
