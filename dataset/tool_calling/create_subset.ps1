# Load JSON file
$json = Get-Content 'synthetic_financial_dataset_20260302.json' -Raw | ConvertFrom-Json

# Create subset with 3 entries per category
$subset = @{}
$json | Get-Member -MemberType NoteProperty | ForEach-Object {
    $key = $_.Name
    $entries = $json.$key
    if ($entries -is [System.Collections.Generic.List`1[System.Object]]) {
        $subset[$key] = @($entries | Select-Object -First 3)
    } else {
        $subset[$key] = $entries
    }
}

# Save subset
$subset | ConvertTo-Json -Depth 100 | Set-Content 'synthetic_financial_dataset_subset.json'

# Print summary
Write-Host "Original dataset:"
$json | Get-Member -MemberType NoteProperty | ForEach-Object {
    $key = $_.Name
    $count = ($json.$key).Count
    Write-Host "  $key`: $count entries"
}

Write-Host ""
Write-Host "Subset dataset (3 entries per category):"
$subset | Get-Member -MemberType NoteProperty | ForEach-Object {
    $key = $_.Name
    $count = ($subset[$key]).Count
    Write-Host "  $key`: $count entries"
}

Write-Host ""
Write-Host "Subset saved to: synthetic_financial_dataset_subset.json"
