function getBathValue() {
    var uiBathrooms = document.getElementsByName('uiBathrooms');
    for (var i in uiBathrooms) {
        if (uiBathrooms[i].checked) {
            return parseInt(i)+1;
        }
    }
    return -1;
}

function getBHKValue(){
    var uiBHK = document.getElementsByName('uiBHK');
    for (var i in uiBHK) {
        if (uiBHK[i].checked) {
            return parseInt(i)+1;
        }
    }
    return -1;
}

function onClickedEstimatePrice(){
    console.log('Estimate price button clicked');

    var sqft = document.getElementById('uiSqft');
    var bhk = getBHKValue();
    var bath = getBathValue();
    var location = document.getElementById('uiLocations')
    var estPrice = document.getElementById('uiEstimatedPrice')

    var url = 'http://127.0.0.1:5000/predict_home_price'; 

    $.post(url, {
        total_sqft: parseFloat(sqft.value),
        bhk: bhk,
        bath: bath,
        location: location.value
    },function(data, status) {
        console.log(data.estimated_price * 100000);
        estPrice.innerHTML = "<h2>" + "Estimated Price:  ₹" + (data.estimated_price * 100000).toString() + "</h2>" + "(Indian Rupees)";
        console.log(status);
    });

}

function onPageLoad(){
    console.log('Document loaded');
    var url = 'http://127.0.0.1:5000/get_location_names';
    $.get(url, function(data, status){
        console.log('got response from get_location_names request');

        if (data){
            var locations = data.locations;
            var uiLocations = document.getElementById('uiLocations');
            $('#uiLocations').empty();

            for (var i in locations) {
                var opt = new Option(locations[i]);
                $('#uiLocations').append(opt);
            }
        }
    });
}

window.onload = onPageLoad;
