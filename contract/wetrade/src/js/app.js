App = {
  web3Provider: null,
  contracts: {},
  account: 0x0,
  accountInfo: null,
  col:"0x74cd927885455a3e13518df69c0fe17b7129c18b",
  loading: false,
  model_adr:'QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco',
  deposit:100000,
  reward:10000000000,
  percentage:10,

  init: async function() { 
    // Load date.
    //App.loadData();
    return await App.initWeb3();
  },

  initWeb3: async function() {
    // Modern dapp browsers...
    if (window.ethereum) {
      App.web3Provider = window.ethereum;
      try {
        // Request account access
        await window.ethereum.enable();
      } catch (error) {
        // User denied account access...
        console.error("User denied account access")
      }
    }
    // Legacy dapp browsers...
    else if (window.web3) {
      App.web3Provider = window.web3.currentProvider;
    }
    // If no injected web3 instance is detected, fall back to Ganache
    else {
      App.web3Provider = new Web3.providers.HttpProvider('http://localhost:8545');
    }
    web3 = new Web3(App.web3Provider);

    console.log(web3);

    //get current account information
    App.displayAccountInfo();
	  
    //display datasets
    App.loadData();
    
    return  App.initContract();
  },

  displayAccountInfo: function() {
    console.log("Display account info");
    web3.eth.getAccounts(function(err, accounts) {
      if (err === null) {
        var account = accounts[0];
        App.account = account;
        
        //console.log(accounts.length);
      	console.log("current account:", App.account, ",  collector:", App.col);
      	if(App.account == App.col.toLowerCase())
      	{
      		console.log("This is the data collector");
      	}
        $("#Account").text(account);
        web3.eth.getBalance(account, function(err, balance) {
          if (err === null) {
            $("#AccountBalance").text(web3.fromWei(balance, "ether") + " ETH");
          }
        });
      }
    });
  },


  loadData: function() {
	
    console.log("[Loading data...]");
	
 
    $.getJSON('../dataset.json', function(data) {
    var datasetsRow = $('#datasetsRow');
    var datasetTemplate = $('#datasetTemplate');

    for (i = 0; i < data.length; i ++) {
      datasetTemplate.find('.panel-title').text(data[i].name);
      datasetTemplate.find('img').attr('src', data[i].picture);
      datasetTemplate.find('.dataset-address').text(data[i].IPFSaddress);
      datasetTemplate.find('.dataset-size').text(data[i].size);
      datasetTemplate.find('.dataset-seller').text(data[i].seller);
      datasetTemplate.find('.btn').attr('data-id', data[i].id);
	
    	if(App.account == App.col.toLowerCase())
    	{	
    		//console.log("[Equal] App.acct", App.account.toString(), "collector", App.col.toString());	
    		datasetTemplate.find('.btn-smpdata').attr('style', "display:true");
    		datasetTemplate.find('.btn-setcandidate').attr('style', "display:true");
    		datasetTemplate.find('.btn-buy').attr('style', "display:true");	

    	}
	
      else if(data[i].seller.toLowerCase() == App.account)
      {	
        /*datasetTemplate.find('.btn-reg').attr('style', "display:true");
        datasetTemplate.find('.btn-reg').attr('disabled',false);
        datasetTemplate.find('.btn-subfeatures').attr('style', "display:true");
        datasetTemplate.find('.btn-subreview').attr('style', "display:true");
        datasetTemplate.find('.btn-clmreward').attr('style', "display:true");
	*/
        datasetTemplate.find('.panel-title').text(data[i].name+"  ***** My Data *****");

        App.accountInfo=data[i];
        //datasetTemplate.find('.btn-buy').hide();
      }
    	else
    	{
    		datasetTemplate.find('.btn').attr('style', "display:none");
    	}

    	datasetsRow.append(datasetTemplate.html());
    }


   if(App.account == App.col.toLowerCase())
    {
	console.log("hhhhhhh");
	$('.btn-reg').hide();
	$('.btn-subfeatures').hide();
	$('.btn-subreview').hide();
    }

    });	
  },

  initContract: function() {
    $.getJSON('Trade.json', function(data) {
      // Get the necessary contract artifact file and use it to instantiate a truffle contract abstraction.
      
      App.contracts.Trade = TruffleContract(data);

      // Set the provider for our contract.
      App.contracts.Trade.setProvider(App.web3Provider);

      console.log("init Contract");

      // Listen for events
      //App.listenToCollectorEvents();

      // Retrieve the dataset from the smart contract
    //	if(App.account == App.col.toLowerCase())
    	//{	
    	//	App.createContract();
    	//}    
      //return App.register(App.account);
      App.bindEvents();

    });
  },

  createContract: function() 
  {
    App.contracts.Trade.deployed().then(function(instance) 
    {
        var trade=instance;

        trade.CreateContract(App.model_adr, App.deposit, App.percentage, {from: App.col.toLowerCase(), value:App.reward, gas:3000000})
        .then(function(tx){
        console.log("##CreateContract: collector set configurations.");

        //show marketplace information
	  	$("#Collector").text(App.col);
	  	$("#Reward").text(App.reward/1000000000+" ETH");
	  	$("#Sample").text(App.percentage+" %");
	  	$("#Least_deposit").text(App.deposit);
	  	$("#Model_address").text(App.model_adr);

      });
    });  
  },

  bindEvents: function() {
    $(document).on('click', '.btn-reg', App.register);
    $(document).on('click', '.btn-smpdata', App.smpdata);
    $(document).on('click', '.btn-setcandidate', App.setcandidata);
    $(document).on('click', '.btn-buy', App.setoptimal);
    //$(document).on('click', '.btn-clmreward', App.clmreward);
  },

  register: function(event) 
  {
    console.log("register");
   
    var adr=App.accountInfo.IPFSaddress;
    var acct=App.accountInfo.seller;

    console.log(adr, "\n", acct);

    App.contracts.Trade.deployed().then(function(instance) 
    {
      var trade=instance;
      //console.log(adr);
      
      trade.Registration(adr, {from: acct, value: 300000,gas:3000000}).then(function(tx)
      {
        alert("#Provider registered.");
        $("#Status").text("Providers Registrating...");
        $('.btn-smpdata').attr('disabled',false);
	$('.btn-reg').attr('disabled',true);
	App.displayAccountInfo();
	App.listenToRegEvents();
      });
    });
  },

  smpdata: async function(event) 
  {
	event.preventDefault();
	var provId = parseInt($(event.target).data('id'));
    
    	console.log("smpdata for provider: #", provId);

	$.getJSON('../dataset.json', function(data) {

	targetPro=data[provId];

  	var adr=targetPro.IPFSaddress;
	var smpsize=Math.ceil(adr.length*App.percentage*0.01);
	var sample_hash=[];

  	for(i=0; i<smpsize; i++)
  	{
  		var ran=Math.floor(Math.random()*(adr.length-i));
  		sample_hash.push(adr[ran]);
  		adr[ran]=adr[adr.length-i-1];
  	}
  	console.log("Sampling address hash:", sample_hash);

  	App.contracts.Trade.deployed().then(function(instance) 
    {
      var trade=instance;
      //console.log(adr);
      
      trade.SampleData(targetPro.seller, sample_hash ,{from: App.col, gas:3000000}).then(function(tx)
      {
        alert("\n##Collector sampling data succeed.");
        $('.btn-subfeatures').attr('disabled',false);
	$('.btn-setcandidate').attr('disabled',false);
        $("#Status").text("Sampling data...");
      });
    });
	});
  },

  subFeatures: function() 
  {
  	var _feature_adr = $("#feature_adr").val();
  	App.contracts.Trade.deployed().then(function(instance) 
    {
      var trade=instance;
      //console.log(adr);
      
      trade.SubmitFeatures(_feature_adr, {from:App.account, gas:3000000}).then(function(tx)
      {
        alert("##Provider submit features succeed.");
	$('.btn-subfeatures').attr('disabled',true);
        $('.btn-setcandidate').attr('disabled',false);
        $("#Status").text("Submitting Features...");
      });
    });
  },

  setcandidata: function(event) 
  {
  	event.preventDefault();
	var provId = parseInt($(event.target).data('id'));
    
    	console.log("SetCandidate, set provider: #", provId);

	$.getJSON('../dataset.json', function(data) {

	targetPro=data[provId];

	var _canditate=targetPro.seller;

  	App.contracts.Trade.deployed().then(function(instance) 
    {
      var trade=instance;
      //console.log(adr);
      
      trade.SetCandidate(_canditate.toLowerCase(), {from:App.col, gas:3000000}).then(function(tx)
      {
        alert("##Collector has set candidate.");

        if(App.account==_canditate.toLowerCase())
        {
        	$('#Candidate').text("Yes! You're");
        	$('.btn-subreview').attr('disabled',false);
        }
        $("#Status").text("Collector has set a candidate!");
        $('#Mark_candidate').text(_canditate);
	$('.btn-buy').attr('disabled',false);
      });
    });
	});

  },

  subReview: function() 
  {
    var _plain_sample_adr=$("#plain_sample_adr").val();
    var _key=$("#key").val();

    App.contracts.Trade.deployed().then(function(instance) 
    {
      var trade=instance;
      //eg. key="0x911e77b214c504666bbd69642887f360310f2fc3a00907a1a32baa07bd0bfe41";
      trade.SubmitReview(App.account, _plain_sample_adr, _key, {from: App.account, gas:3000000}).then(function(tx)
      {
        alert("##Provider submit review succeed.");
        $("#Status").text("Candidate has submit Review!");
      });
    });      

  },

  setoptimal: function(event) 
  {
  	event.preventDefault();
	var provId = parseInt($(event.target).data('id'));
    
    	console.log("SetOptimal, set provider: #", provId);

	$.getJSON('../dataset.json', function(data) {

	var targetPro=data[provId];

	var _optimal=targetPro.seller;

  	App.contracts.Trade.deployed().then(function(instance) 
    {
      var trade=instance;
 
      trade.SetOptimal(_optimal.toLowerCase(), {from:App.col, gas:3000000}).then(function(tx)
      {
        alert("##Collector has set optimal.");
        $('#Mark_optimal').text(_optimal);
        if(App.account==_optimal.toLowerCase())
        {
        	$('#Optimal').text("Yes! You're");
        	$('#datasetsRow').find('.btn-clmreward').attr('disabled',false);
        }
        $("#Status").text("Collector has set an optimal!");
	alert("Trade Success!");
      });
	});
    });

  },

  clmReward: function() 
  {
  	var _plain_adr=$("#plain_adr").val();

  	App.contracts.Trade.deployed().then(function(instance) 
    {
      var trade=instance;
      
      trade.ClaimReward(_plain_adr, {from: App.account, gas:3000000}).then(function(tx)
      {
        alert("##Provider submit all plain data address succeed.");
        App.displayAccountInfo();
        $("#Status").text("Deal Done!");
      });
    });   
  },


  displayDataset: function(id, seller, name, description, dataset_adr) {
    // Retrieve the dataset placeholder
    var datasetsRow = $('#datasetsRow');

    //var etherPrice = web3.fromWei(price, "ether");

    // Retrieve and fill the dataset template
    var datasetTemplate = $('#datasetTemplate');
    datasetTemplate.find('.panel-title').text(name);
    datasetTemplate.find('.dataset-description').text(description);
    datasetTemplate.find('.dataset-address').text(dataset_adr);
    datasetTemplate.find('.btn-buy').attr('data-id', id);
    //datasetTemplate.find('.btn-buy').attr('data-value', etherPrice);

    // seller?
    if (seller == App.account) {
      datasetTemplate.find('.dataset-seller').text("You");
      datasetTemplate.find('.btn-buy').hide();
    } else {
      datasetTemplate.find('.dataset-seller').text(seller);
      datasetTemplate.find('.btn-buy').show();
    }

    // add this new dataset
    datasetsRow.append(datasetTemplate.html());
  },

  sellDataset: function() {
    // retrieve details of the dataset
    var _dataset_name = $("#dataset_name").val();
    var _description = $("#dataset_description").val();
    var _address = $("#dataset_address").val();

    console.log(_address);

    if (_address == '') {
      console.log("nothing to sell");
      return false;
    }

    App.contracts.Trade.deployed().then(function(instance) {
      return instance.Registration(_address, {from: App.account, value: 300000, gas:3000000});
    }).then(function(result) {
      console.log("Registration");
      console.log(App.account);

    }).catch(function(err) {
      console.error(err);
    });
  },

  // Listen for events raised from the contract
listenToRegEvents: function() 
  {
	App.contracts.Trade.deployed().then(function(instance) {
	var event1=instance.Evt_Regst({},{});

	console.log("event1: ",event1);
	event1.watch(function(error, result){
	if(!error)
	{
	    console.log("....Listen....Receive an Registration....");
		console.log("result.args",result.args);
		var pid=result.args.pid;
		var data_hash=result.args.hashes;
		console.log("....pid and data hashes are...\n", pid, data_hash);
	}else{
	console.log("error:", error);
	}
	});
	
	var rlt1=event1.get(function(error, logs){
		console.log("get"+ JSON.stringify(logs.args));
	});


	});

	
}


};

//$(function() {
  $(window).on("load",function() {
    console.log("load");
    App.init();
  });
//});



