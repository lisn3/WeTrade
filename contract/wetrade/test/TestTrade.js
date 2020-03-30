var Trade=artifacts.require("./Trade.sol")



contract('Trade',function(accounts){



  var acct=accounts[5];  //optimal provider

  var col=accounts[0];

  

  it("should successfully exchange",function(){

    return Trade.deployed().then(function(instance){

      trade=instance;

      var model_adr='QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco';

      var deposit=10000;

      var percentage=10;



      return trade.CreateContract(model_adr, deposit, percentage,{from: col, value:1000000000}).then(function(tx){

        console.log("##CreateContract: collector set configurations. gas Cost is: "+tx.receipt.gasUsed);



        return web3.eth.getBalance(acct);}).then((dpst_p1)=>{

        console.log("Before provider register，his balance is :"+ dpst_p1); 
        var data_hash=new Array(1000);
        const hash="0xaf230a56630ab5f081d5fd9d4c4c08c65417e0e49b680b87d287923aa6c025bd";
        data_hash.fill(hash)
        return trade.Registration(data_hash, {from: acct, value: 300000}).then(function(tx){

          console.log("##Registration: provider "+5+" register to contract. gas Cost is: "+tx.receipt.gasUsed);

          

          return web3.eth.getBalance(acct);}).then((dpst_p)=>{

          console.log("After provider 5 register，his balance is :"+ dpst_p);


          const hash="0xaf230a56630ab5f081d5fd9d4c4c08c65417e0e49b680b87d287923aa6c025bd";

          var sample_hash=new Array(100);  //sample 20 blocks;

          sample_hash.fill(hash);

          return trade.SampleData(acct, sample_hash ,{from: col}).then(function(tx){

          console.log("\n##SampleData: collector sample trader "+5+" data hash. gas Cost is: "+tx.receipt.gasUsed);


          return trade.SubmitFeatures("QmXoypizjW3WknFiJnKLwHCnL66vedxjQkDDP1mXWo4uco", {from:acct}).then(function(tx){

          console.log("##SubmitFeatures: provider 5 upload intermediate features of sampled data. gas Cost is: "+tx.receipt.gasUsed);

          console.log("\n..........Evaluation Ofchain...........");



          return trade.SetCandidate(acct, {from:col}).then(function(tx){

          console.log("\n##SetCandidate: collector set a candidate trader 5. gas cost is: "+tx.receipt.gasUsed);

          
          const plain="QmXoypizjW3WknFiJnKLwHCnL66vedxjQkDDP1mXWo6uco";
          var plain_sample_adr="";

          var rpt=100;
          for(var j=0;j<rpt;j++){
            plain_sample_adr+=plain;
          }

          var key="0x911e77b214c504666bbd69642887f360310f2fc3a00907a1a32baa07bd0bfe41";

          return trade.SubmitReview(acct, plain_sample_adr, key, {from: acct}).then(function(tx){

          console.log("\n##SubmitReview: Provider 5 upload plain sample address and his session key. gas Cost is: "+tx.receipt.gasUsed);

          return trade.SetOptimal(acct, {from:col}).then(function(tx){

          console.log("##SetOptimal: collector set an optimal data provider. gas Cost is: "+tx.receipt.gasUsed);


          const plain="QmXoypizjW3WknFiJnKLwHCnL66vedxjQkDDP1mXWo6uco";
          //var data_plain="QmXoypizjW3WknFiJnKLwHCnL66vedxjQkDDP1mXWo6ucoQmXoypizjW3WknFiJnKLwHCnL66vedxjQkDDP1mXWo5ucoQmXoypizjW3WknFiJnKLwHCnL66vedxjQkDDP1mXWo6ucoQmXoypizjW3WknFiJnKLwHCnL66vedxjQkDDP1mXWo6ucoQmXoypizjW3WknFiJnKLwHCnL66vedxjQkDDP1mXWo6ucoQmXoypizjW3WknFiJnKLwHCnL66vedxjQkDDP1mXWo6ucoQmXoypizjW3WknFiJnKLwHCnL66vedxjQkDDP1mXWo6ucoQmXoypizjW3WknFiJnKLwHCnL66vedxjQkDDP1mXWo6ucoQmXoypizjW3WknFiJnKLwHCnL66vedxjQkDDP1mXWo6ucoQmXoypizjW3WknFiJnKLwHCnL66vedxjQkDDP1mXWo6uco";  //plain IPFS address according to the data_hash.
          //var data_plain has a different address at index 2 (end with 5uco)
          var data_plains="";
          var rpt=1000;
 	        for(var i=0;i<rpt;i++){
            data_plains+=plain;
          }

          return trade.ClaimReward(data_plains, {from: acct}).then(function(tx){

          console.log("\n##ClaimReward: optimal provider reveals all plain data address. gas Cost is: "+tx.receipt.gasUsed);            

          

          return web3.eth.getBalance(acct);}).then((dest_af)=>{



          console.log("After reveal, the balance of optimal provider is : "+ dest_af);

          

          return trade.ClaimDeposit({from:acct}).then(function(tx){

          console.log("##ClaimDeposit: traders, gas cost: "+tx.receipt.gasUsed);



          return web3.eth.getBalance(acct);}).then((fp)=>{

            console.log("After optimal provider refund, his balance is : "+fp);

          });      

        });

      });

    });});});});

  });});}); });});






